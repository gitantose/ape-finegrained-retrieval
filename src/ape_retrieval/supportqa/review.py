"""Keyboard-only SupportQA human evaluation. Run with --help for commands."""

import argparse
from copy import deepcopy
import curses
import fcntl
import getpass
import hashlib
from pathlib import Path
import textwrap

from ape_retrieval.encode import atomic_json
from ape_retrieval.supportqa.human_review import (
    CAUSES, MODELS, ROOT, TERMINAL, Study, blank, current_version, inline_judgment,
    new_draft, now, open_queues, prepare, read, save_version, state, verify_hashes,
)


HELP = [
    'Enter a judgment to save and advance.',
    '2 = proposition 2 alone. 2 4 = both required. 2 / 4 = either alone.',
    'm missing evidence   d distorted evidence   i insufficient source',
    'a ambiguous   r reference answer objection   u uncertain',
    'r and u stay pending review.',
    ':spans shows numbered evidence spans and offsets.',
    ':e 1 3 selects evidence spans. :e alone clears the selection.',
    ':n short note adds an optional note. :n alone clears it.',
    ':skip leaves a case unfinished. :prev, :next and :goto 12 revisit.',
    ':undo reverses the latest save in this queue.',
    ':compare opens the audit after a save. Evaluation and second review',
    'open comparisons after every case has a saved decision.',
    ':cause 1 4 | note records diagnosis in comparison.',
    'Causes: 1 decomposition, 2 label, 3 matching, 4 ranking,',
    '5 suspected representation, 6 unresolved.',
    ':back returns to the question. :q exits. Scroll with arrows or PgUp/PgDn.',
]


class Records:
    def __init__(self, root, queue, reviewer):
        role = 'independent' if queue == 'second' else 'primary'
        self.directory = root / role / queue
        self.directory.mkdir(parents=True, exist_ok=True)
        self.profile = root / role / 'reviewer.json'
        self.reviewer = {'id': reviewer, 'role': role}
        other_role = 'primary' if role == 'independent' else 'independent'
        other = root / other_role / 'reviewer.json'
        if other.exists() and read(other)['id'] == reviewer:
            raise ValueError('The independent and primary reviewers need different IDs.')
        if self.profile.exists() and read(self.profile) != self.reviewer:
            raise ValueError(f'This role is already registered to a different reviewer in {self.profile}.')
        if not self.profile.exists():
            atomic_json(self.profile, self.reviewer)

    def path(self, qid):
        return self.directory / (hashlib.sha256(qid.encode()).hexdigest() + '.json')

    def load(self, qid):
        path = self.path(qid)
        record = read(path) if path.exists() else blank(qid)
        if record['query_id'] != qid:
            raise ValueError(f'Judgment file has the wrong query ID: {path}')
        return record

    def save(self, record):
        atomic_json(self.path(record['query_id']), record)


def comparison_allowed(queue, record, records):
    return state(record) in TERMINAL and (
        queue == 'diagnostic' or all(state(item) in TERMINAL for item in records))


def evidence_lines(evidence, record, show_spans=False):
    saved = current_version(record)
    judgment = record['draft'] or (saved['judgment'] if saved else new_draft())
    selected = {span['sentence'] for span in judgment['source']['spans']}
    lines = ['Question: ' + evidence['question'],
             'Dataset reference answer: ' + (' | '.join(evidence['answers']) or '(none)'), '',
             'Source spans' if show_spans else 'Context']
    if show_spans:
        for sentence in evidence['sentences']:
            number = sentence['id']
            mark = '*' if number in selected else ' '
            lines.append(f'{mark}{number}. [{sentence["start"]}:{sentence["end"]}] {sentence["text"]}')
    else:
        lines.extend(evidence['context'].splitlines())
    lines.extend(['', 'Generated propositions'])
    lines.extend(f'{i}. [{pid}] {text}' for i, (pid, text) in enumerate(evidence['propositions'].items(), 1))
    if not evidence['propositions']:
        lines.append('(No saved propositions)')
    if judgment['explanation']:
        lines.append('Note: ' + judgment['explanation'])
    if saved:
        decision = saved['judgment']
        outcome = decision.get('pending_reason') or decision['source']['answerability']
        support = ' / '.join(' '.join(group) for group in decision['support_sets'])
        flags = ', '.join(decision['defects'])
        lines.append('Saved: ' + ', '.join(part for part in (outcome, support, flags) if part))
    return lines


def comparison_lines(study, qid, record, selection):
    study.load_rankings()
    silver = study.silver[qid]
    version = current_version(record)
    human = version['judgment']
    match = study.matching(qid)
    human_sets = {tuple(sorted(group)) for group in human['support_sets']}
    silver_sets = {tuple(sorted(group)) for group in silver['support_sets']}
    lines = [
        study.queries[qid]['query'], '',
        f'Silver outcome: {silver["support_type"]}. Source: {silver["document_answerability"]}.',
        f'Human decision v{version["version"]}: {version["status"]}. Class: {version["support_class"]}.',
        f'Silver support sets: {silver["support_sets"]}', f'Human support sets: {human["support_sets"]}',
        f'Added alternatives: {sorted(human_sets - silver_sets)}',
        f'Removed alternatives: {sorted(silver_sets - human_sets)}', '',
        'Saved APE alignment: ' + match.get('status', 'unavailable'),
    ]
    if qid in selection.get('reasons', {}):
        lines.insert(2, 'Diagnostic selection: ' + selection['reasons'][qid])
    if qid in selection.get('cases', {}):
        lines.insert(2, 'Sampling stratum: ' + ' / '.join(selection['cases'][qid]['stratum']))
    if match.get('reason'):
        lines.append('Alignment exclusion: ' + match['reason'])
    for pair in match.get('pairs', []):
        lines.append(f'{pair["reference_id"]} → slot {pair["slot_id"]} (cosine {pair["cosine"]:.4f})')
    lines.append(f'Unmatched propositions: {match.get("unmatched_reference_ids", "unavailable")}')
    if 'unmatched_slot_ids' in match:
        lines.append(f'Unmatched slots: {match["unmatched_slot_ids"]}')
    if match.get('overflow_reference_ids'):
        lines.append(f'Overflow references: {match["overflow_reference_ids"]}')
    if match.get('cosine'):
        lines.append('Saved similarities, columns by slot: ' + ', '.join(map(str, match['slot_ids'])))
        lines.extend(pid + '  ' + ' '.join(f'{value:.4f}' for value in row)
                     for pid, row in zip(match['reference_ids'], match['cosine'], strict=True))
    lines.extend(['', 'Unmatched means the saved assignment did not link the reference. '
                  'The assignment alone cannot establish what a slot represents.', '', 'Saved ranks, best first:'])
    for model in MODELS:
        row = study.ranks[model][qid]
        ranked = row.get('ranked_ids')
        label = 'slot ' if model == 'ape' else ''
        lines.append(f'{model}: ' + (', '.join(f'{i}. {label}{pid}' for i, pid in enumerate(ranked, 1))
                                     if ranked else f'unavailable ({row["eligibility"]})'))
        if ranked and row['eligibility'] != 'eligible':
            lines.append(f'  Original scoring exclusion: {row["eligibility"]}')
    lines.extend(['', 'Source text and propositions', *evidence_lines(study.evidence(qid), record)])
    for diagnosis in record['diagnoses']:
        lines.append(f'Diagnosis for v{diagnosis["judgment_version"]}: {", ".join(diagnosis["causes"])}. {diagnosis["note"]}')
    return lines


class Review:
    def __init__(self, screen, study, store, queue, selection, root):
        self.screen, self.study, self.store, self.queue = screen, study, store, queue
        self.selection, self.root = selection, root
        self.ids = selection['query_ids']
        self.records = [store.load(qid) for qid in self.ids]
        self.position_path = store.directory / 'position.json'
        self.buffer, self.message = '', ''
        self.visit(read(self.position_path)['index'] if self.position_path.exists() else 0)

    def visit(self, position):
        self.position = position % len(self.ids)
        self.record = self.records[self.position]
        self.evidence = self.study.evidence(self.record['query_id'])
        self.page, self.scroll = 'question', 0
        if not self.position_path.exists() or read(self.position_path)['index'] != self.position:
            atomic_json(self.position_path, {'index': self.position})

    def write(self, row, content):
        height, width = self.screen.getmaxyx()
        if 0 <= row < height and width > 1:
            line = ''.join(c if c.isprintable() else ' ' for c in content)
            try:
                self.screen.addnstr(row, 0, line, width - 1)
            except curses.error:
                # A concurrent resize or a wide character can reach the screen edge.
                pass

    def render(self):
        screen = self.screen
        screen.erase()
        height, width = screen.getmaxyx()
        if height < 8 or width < 28:
            self.write(0, 'Enlarge terminal (:q exits).')
            screen.refresh()
            return
        states = [state(record) for record in self.records]
        count = states.count('confirmed')
        pending = states.count('needs adjudication')
        status = state(self.record).replace('needs adjudication', 'pending review')
        self.write(0, f'{self.queue.title()} {self.position + 1}/{len(self.ids)}   '
                      f'{count} confirmed · {pending} pending   This case: {status}')
        if self.page == 'help':
            lines = HELP
        elif self.page == 'comparison':
            lines = comparison_lines(self.study, self.record['query_id'], self.record, self.selection)
        else:
            lines = evidence_lines(self.evidence, self.record, show_spans=self.page == 'spans')
        wrapped = [part for line in lines for part in (textwrap.wrap(line, width - 2) or [''])]
        visible = height - 6
        self.scroll = min(self.scroll, max(0, len(wrapped) - visible))
        for number, line in enumerate(wrapped[self.scroll:self.scroll + visible], 1):
            self.write(number, line)
        self.write(height - 5, 'Enter saves  2 alone  2 4 together  2 / 4 either')
        self.write(height - 4, 'm missing d distorted i insufficient a ambiguous r objection u unsure')
        self.write(height - 3, '↑↓ PgUp/PgDn scroll  :spans :skip :prev :undo :compare :help :q')
        self.write(height - 2, self.message or f'Lines {self.scroll + 1}–{min(self.scroll + visible, len(wrapped))}/{len(wrapped)}')
        self.write(height - 1, '> ' + self.buffer[-(width - 4):])
        screen.refresh()

    def editable(self):
        saved = current_version(self.record)
        return deepcopy(self.record['draft'] or (saved['judgment'] if saved else new_draft()))

    def save_judgment(self, value):
        updated = deepcopy(self.record)
        updated['draft'], status = inline_judgment(value, self.evidence, self.editable())
        save_version(updated, self.evidence, self.store.reviewer, status)
        self.store.save(updated)
        self.record.update(updated)
        self.visit(self.position + 1)
        self.message = 'Saved ' + ('confirmed.' if status == 'confirmed' else 'pending review.')

    def undo(self):
        saves = [(v['saved_at'], i, v) for i, record in enumerate(self.records)
                 if (v := current_version(record)) and v['judgment'].get('entry_mode') == 'inline']
        if not saves:
            raise ValueError('No saved entry to undo in this queue.')
        _, i, version = max(saves)
        version['undone_at'] = now()
        self.store.save(self.records[i])
        self.visit(i)
        self.message = 'Last save undone. Enter a replacement when ready.'

    def optional(self, command, value):
        judgment = self.editable()
        if command == ':e':
            sentences = {row['id']: row for row in self.evidence['sentences']}
            ids = sorted(set(map(int, value.split())))
            if any(i not in sentences for i in ids):
                raise ValueError('Use source sentence numbers shown above.')
            judgment['source']['spans'] = [{'sentence': i, 'start': sentences[i]['start'],
                                            'end': sentences[i]['end']} for i in ids]
        else:
            judgment['explanation'] = value
        self.record['draft'] = judgment
        self.store.save(self.record)
        self.message = 'Optional detail saved with the draft.'

    def diagnose(self, value):
        choices, separator, note = value.partition('|')
        ids = sorted(set(map(int, choices.split())))
        if not ids or any(not 1 <= i <= len(CAUSES) for i in ids) or not separator or not note.strip():
            raise ValueError('Use :cause 1 4 | note. See :help for cause numbers.')
        self.record['diagnoses'].append({'causes': [CAUSES[i - 1] for i in ids],
                                         'note': note.strip(), 'reviewer': self.store.reviewer,
                                         'saved_at': now(),
                                         'judgment_version': current_version(self.record)['version']})
        self.store.save(self.record)
        self.message = 'Diagnosis saved.'

    def submit(self, value):
        command, _, argument = value.partition(' ')
        argument = argument.strip()
        if command in (':prev', ':next'):
            self.visit(self.position + (1 if command == ':next' else -1))
        elif command == ':goto':
            index = int(argument) - 1
            if not 0 <= index < len(self.ids):
                raise ValueError('Case number is outside this queue.')
            self.visit(index)
        elif command == ':skip':
            if state(self.record) not in TERMINAL:
                self.record['skipped'] = True
                self.store.save(self.record)
            self.visit(self.position + 1)
        elif command == ':undo':
            self.undo()
        elif command in (':help', ':back'):
            self.page, self.scroll = ('help' if command == ':help' else 'question'), 0
        elif command == ':spans':
            self.page, self.scroll = 'spans', 0
            self.message = ':e NUMBERS selects spans. :back returns to the context.'
        elif command == ':compare':
            if not comparison_allowed(self.queue, self.record, self.records):
                raise ValueError('Save this judgment first. Evaluation comparisons open after the full queue.')
            verify_hashes(read(self.root / 'diagnostic.json')['files'])
            self.study.load_rankings()
            self.record['comparison_seen'] = self.record['comparison_seen'] or now()
            self.store.save(self.record)
            self.page, self.scroll = 'comparison', 0
            self.message = ':cause NUMBERS | note saves a diagnosis. :back returns.'
        elif self.page == 'comparison' and command == ':cause':
            self.diagnose(argument)
        elif self.page in ('question', 'spans') and command in (':e', ':n'):
            self.optional(command, argument)
        elif self.page == 'question' and value:
            self.save_judgment(value)
        elif value:
            raise ValueError('Use :back to return to the question, or :help for commands.')

    def run(self):
        while True:
            self.render()
            key = self.screen.get_wch()
            if key in ('\n', '\r', curses.KEY_ENTER):
                value, self.buffer = self.buffer.strip(), ''
                if value == ':q':
                    return
                self.message = ''
                try:
                    self.submit(value)
                except (ValueError, OverflowError) as exc:
                    self.message = str(exc)
            elif key in (curses.KEY_BACKSPACE, '\b', '\x7f'):
                self.buffer = self.buffer[:-1]
            elif key == '\x1b':
                self.buffer = ''
            elif key in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_PPAGE, curses.KEY_NPAGE):
                step = max(1, self.screen.getmaxyx()[0] - 6)
                delta = {curses.KEY_UP: -1, curses.KEY_DOWN: 1, curses.KEY_PPAGE: -step, curses.KEY_NPAGE: step}
                self.scroll = max(0, self.scroll + delta[key])
            elif isinstance(key, str) and key.isprintable():
                self.buffer += key


def review_queue(study, root, queue, reviewer):
    selected = open_queues(root, queue)
    role = 'independent' if queue == 'second' else 'primary'
    with (root / f'{role}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Records(root, queue, reviewer)
        curses.wrapper(lambda screen: Review(screen, study, store, queue, selected, root).run())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'evaluation', 'diagnostic', 'second', 'report', 'reconcile'))
    parser.add_argument('--reviewer', help='Local provenance ID; required for the independent reviewer.')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    study = Study()
    try:
        if args.command == 'prepare':
            args.root.mkdir(parents=True, exist_ok=True)
            with (args.root / 'prepare.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                prepare(study, args.root)
        elif args.command in ('report', 'reconcile'):
            from ape_retrieval.supportqa.review_report import report, reconcile
            (report if args.command == 'report' else reconcile)(study, args.root)
        else:
            if args.command == 'second' and not args.reviewer:
                raise ValueError('Pass --reviewer with the independent reviewer’s ID.')
            review_queue(study, args.root, args.command, args.reviewer or getpass.getuser())
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
