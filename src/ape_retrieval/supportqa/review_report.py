"""Validate the local judgment layer and score it against frozen SupportQA ranks."""

from collections import Counter

from ape_retrieval.propsearch import sha256
from ape_retrieval.supportqa.human_review import (
    CAUSES, KS, MODELS, TERMINAL, current_version, open_queues, read, scored, state, support_class,
    validate_decision, validate_sets, validate_source, verify_hashes,
)
from ape_retrieval.supportqa.review import Records


METRICS = ('first_support_reciprocal_rank', *(f'Recall@{k}' for k in KS),
           *(f'all_required_Success@{k}' for k in KS))


def judgment_records(study, root, queue, ids):
    role = 'independent' if queue == 'second' else 'primary'
    profile = root / role / 'reviewer.json'
    if not profile.exists():
        if (root / role / queue).exists() and any((root / role / queue).glob('*.json')):
            raise ValueError(f'Missing reviewer provenance for {queue}')
        return {}
    reviewer = read(profile)
    store = Records(root, queue, reviewer['id'])
    paths = {store.path(qid) for qid in ids}
    if set(store.directory.glob('*.json')) - paths - {store.directory / 'position.json'}:
        raise ValueError(f'Judgment records outside the frozen {queue} queue.')
    records = {}
    for qid in ids:
        if not store.path(qid).exists():
            continue
        record = store.load(qid)
        initial = record['initial']
        seen = record['comparison_seen']
        if initial and (initial['reviewer'] != reviewer or
                        record['reference_seen'] and initial['saved_at'] > record['reference_seen']):
            raise ValueError(f'Invalid initial judgment provenance for {qid}')
        if record['reference_seen'] and not initial:
            raise ValueError(f'Reference shown before the initial judgment: {qid}')
        evidence = study.evidence(qid)
        if initial:
            validate_source(initial['source'], evidence)
        if record['draft'] is not None:
            validate_source(record['draft']['source'], evidence, complete=False)
            validate_sets(record['draft']['support_sets'], evidence['propositions'].keys())
        for i, version in enumerate(record['versions']):
            confirmed = version['status'] == 'confirmed'
            inline = version['judgment'].get('entry_mode') == 'inline'
            if (version['version'] != i + 1 or version['status'] not in TERMINAL
                    or version['reviewer'] != reviewer
                    or (not inline and (not record['reference_seen'] or version['saved_at'] < record['reference_seen']))
                    or (version.get('undone_at') and (not inline or version['undone_at'] < version['saved_at']))
                    or version['support_class'] != support_class(version['judgment'])
                    or (version['comparison_seen'] and (not seen or version['saved_at'] < seen))
                    or (seen and version['saved_at'] > seen and not version['comparison_seen'])):
                raise ValueError(f'Invalid saved version provenance for {qid}')
            validate_decision({**record, 'versions': record['versions'][:i]},
                              version['judgment'], evidence, confirmed)
        if seen and (not record['versions'] or record['versions'][0]['saved_at'] > seen
                     or record['versions'][0]['comparison_seen']):
            raise ValueError(f'Comparison shown before a saved judgment: {qid}')
        for diagnosis in record['diagnoses']:
            if (not seen or diagnosis['saved_at'] < seen or diagnosis['reviewer'] != reviewer
                    or not 1 <= diagnosis['judgment_version'] <= len(record['versions'])
                    or not diagnosis['causes'] or set(diagnosis['causes']) - set(CAUSES)
                    or not diagnosis['note'].strip()):
                raise ValueError(f'Invalid saved diagnosis for {qid}')
        records[qid] = record
    return records


def latest_confirmations(records, blinded):
    confirmed = {}
    for qid, record in records.items():
        current = current_version(record)
        if not current or current['status'] != 'confirmed':
            continue
        versions = [v for v in record['versions']
                    if not v.get('undone_at') and (not blinded or not v['comparison_seen'])]
        if versions and versions[-1]['status'] == 'confirmed':
            confirmed[qid] = versions[-1]['judgment']
    return confirmed


def progress(queue, ids, records):
    states = Counter(state(row) for row in records.values())
    states['unreviewed'] += len(ids) - len(records)
    counts = ', '.join(f'{states[name]} {name.replace("needs adjudication", "pending review")}'
                       for name in ('unreviewed', 'draft', *TERMINAL))
    print(f'\n## {queue.title()}\n\n{len(ids)} queued. {counts}.')
    skipped = sum(row['skipped'] for row in records.values())
    post = sum(any(v['comparison_seen'] for v in row['versions']) for row in records.values())
    print(f'{skipped} marked skipped. {post} cases with a saved post-comparison revision.')
    pending = Counter(v['judgment'].get('pending_reason') or 'legacy adjudication'
                      for row in records.values() if state(row) == 'needs adjudication'
                      and (v := current_version(row)))
    if pending:
        print('Pending reasons: ' + ', '.join(f'{reason} ({count})' for reason, count in sorted(pending.items())))


def support_ids(sets):
    return {frozenset(group) for group in sets}


def human_silver_changes(study, confirmed):
    transitions = {name: Counter() for name in ('Answerability', 'Sufficient proposition coverage',
                                                'Support class', 'Support sets')}
    for qid, human in confirmed.items():
        silver = study.silver[qid]
        old_answer = {'insufficient_document': 'insufficient'}.get(
            silver['document_answerability'], silver['document_answerability'])
        transitions['Answerability'][old_answer, human['source']['answerability']] += 1
        old_sets, new_sets = support_ids(silver['support_sets']), support_ids(human['support_sets'])
        transitions['Sufficient proposition coverage'][str(bool(old_sets)), str(bool(new_sets))] += 1
        transitions['Support class'][support_class({'support_sets': silver['support_sets']}), support_class(human)] += 1
        change = ('unchanged' if old_sets == new_sets else
                  'added only' if old_sets < new_sets else 'removed only' if new_sets < old_sets else 'added and removed')
        transitions['Support sets']['silver', change] += 1
    n = len(confirmed)
    print(f'\nHuman versus silver on the same {n} confirmed query IDs')
    print('\n| Dimension | Silver | Human or change | Queries |\n|---|---|---|---:|')
    for dimension, counts in transitions.items():
        for (a, b), count in sorted(counts.items()):
            print(f'| {dimension} | {a} | {b} | {count} |')
    flags = Counter(flag for case in confirmed.values() for flag in case['defects'])
    print(f'\nHuman coverage flags among {n}: {flags["missing"]} missing, {flags["distorted"]} distorted. Flags may overlap.')


def model_scores(study, confirmed):
    study.load_rankings()
    human = {m: {} for m in MODELS}
    silver = {m: {} for m in MODELS}
    eligibility = {m: Counter() for m in MODELS}
    newly_unavailable = {m: Counter() for m in MODELS}
    for qid, judgment in confirmed.items():
        for model in MODELS:
            case = study.silver[qid]
            human_metrics, reason = scored(study, qid, model, judgment['support_sets'])
            silver_metrics, silver_reason = scored(study, qid, model, case['support_sets'])
            if silver_metrics is not None:
                silver[model][qid] = silver_metrics['best_sufficient_set_per_metric']
            if human_metrics is not None:
                human[model][qid] = human_metrics['best_sufficient_set_per_metric']
            elif judgment['support_sets'] and not case['support_sets']:
                newly_unavailable[model][reason] += 1
            original = study.ranks[model][qid]['eligibility']
            eligibility[model][original, reason or 'eligible'] += 1
            if (silver_metrics is not None) != (original == 'eligible'):
                raise ValueError(f'Scoring did not reproduce frozen eligibility for {qid} / {model}: {silver_reason}')
    return human, silver, eligibility, newly_unavailable


def common(scores):
    return set.intersection(*(set(scores[m]) for m in MODELS))


def metric_table(title, tables, total):
    print(f'\n### {title}\n\nCoverage uses {total} confirmed queries as its denominator.')
    keys = ['MRR', *(f'R@{k}' for k in KS), *(f'All S@{k}' for k in KS)]
    print('\n| Model | Labels | Scored / confirmed | ' + ' | '.join(keys) + ' |')
    print('|---|---|---:|' + '---:|' * len(keys))
    for model, label, scores, ids in tables:
        ids = sorted(ids)
        values = [f'{sum(scores[q][metric] for q in ids) / len(ids):.4f}' if ids else '—'
                  for metric in METRICS]
        print(f'| {model} | {label} | {len(ids)}/{total} | ' + ' | '.join(values) + ' |')


def print_scores(study, confirmed):
    human, silver, changes, missing = model_scores(study, confirmed)
    n = len(confirmed)
    h_common, s_common = common(human), common(silver)
    paired = h_common & s_common
    print(f'\nFour-model eligibility on {n} confirmed queries: {len(h_common)} human, {len(s_common)} silver, '
          f'{len(paired)} paired. {len(h_common - s_common)} gained and {len(s_common - h_common)} lost.')
    print('\n| Model | Human eligible / confirmed | Silver eligible / confirmed | Paired | Newly supported with unavailable ranks or assignment |')
    print('|---|---:|---:|---:|---:|')
    for m in MODELS:
        print(f'| {m} | {len(human[m])}/{n} | {len(silver[m])}/{n} | {len(human[m].keys() & silver[m].keys())} '
              f'| {missing[m].total()} |')
    print('\n| Model | Frozen eligibility | Human eligibility | Queries |\n|---|---|---|---:|')
    for m in MODELS:
        for (before, after), count in sorted(changes[m].items()):
            print(f'| {m} | {before} | {after} | {count} |')
    if any(missing.values()):
        print('\nNewly supported with unavailable evidence by model:')
        for model, reasons in missing.items():
            for reason, count in reasons.items():
                print(f'- {model}: {count} {reason}')
    metric_table('Human-confirmed metrics, each model’s available rankings',
                 [(m, 'human', human[m], human[m].keys()) for m in MODELS], n)
    metric_table('Human and silver metrics, paired within each model',
                 [(m, label, scores[m], human[m].keys() & silver[m].keys())
                  for m in MODELS for label, scores in (('human', human), ('silver', silver))], n)
    metric_table('Four-model matched metrics on identical human and silver query IDs',
                 [(m, label, scores[m], paired)
                  for m in MODELS for label, scores in (('human', human), ('silver', silver))], n)
    if h_common != paired:
        metric_table('Human metrics on all four-model eligible human query IDs',
                     [(m, 'human', human[m], h_common) for m in MODELS], n)
    return h_common


def population(study, selection, confirmed, common_ids):
    strata = {(row['source'], row['outcome']): row for row in selection['strata']}
    confirmed_strata = Counter(tuple(selection['cases'][qid]['stratum']) for qid in confirmed)
    print('\n### Coverage by sampling stratum\n')
    print('| Source | Frozen silver outcome | Population | Drawn | Confirmed | Inclusion probability | Weight |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for key, row in sorted(strata.items()):
        print(f'| {key[0]} | {key[1]} | {row["population"]} | {row["sample"]} | {confirmed_strata[key]} '
              f'| {row["probability"]:.6f} | {row["weight"]:.3f} |')
    print('\nUse inverse inclusion weights N_h/n_h for population estimates. For a conditional mean, divide '
          'the weighted sum of eligible scores by the sum of their eligible weights. '
          'The metric tables above are unweighted descriptions of the reviewed sample.')
    if len(confirmed) != len(selection['query_ids']):
        print('Population estimates are withheld until all sampled queries have blinded confirmations. '
              'Partial responses may differ systematically.')
        return
    weights = {q: strata[tuple(selection['cases'][q]['stratum'])]['weight'] for q in confirmed}
    total = sum(weights.values())
    outcomes = {'Source answerable': lambda q: confirmed[q]['source']['answerability'] == 'answerable',
                'Has sufficient generated support': lambda q: bool(confirmed[q]['support_sets']),
                'Same silver/human sufficient sets':
                    lambda q: support_ids(confirmed[q]['support_sets']) == support_ids(study.silver[q]['support_sets']),
                'Human-score eligible for all four models': lambda q: q in common_ids}
    print(f'\nDesign-weighted query estimates from {len(confirmed)} confirmed cases (weight sum {total:,.0f}):')
    for label, predicate in outcomes.items():
        print(f'- {label}: {sum(w for q, w in weights.items() if predicate(q)) / total:.2%}')


def final_report(study, root, audit_path):
    audit = read(audit_path)
    verify_hashes(audit['input_sha256'])
    if sha256(audit_path.with_name('new-predictions.json')) != audit['new_predictions_sha256']:
        raise ValueError('Final audit prediction checksum mismatch.')
    selection = open_queues(root, 'evaluation')
    records = judgment_records(study, root, 'evaluation', selection['query_ids'])
    confirmed = latest_confirmations(records, blinded=True)
    supported = {qid for qid, judgment in confirmed.items() if judgment['support_sets']}
    cohort, queries = audit['cohort'], audit['queries']
    matched = cohort['matched']
    if (audit['shared_cutoffs'] != list(KS) or set(cohort['confirmed']) != confirmed.keys()
            or set(cohort['supported']) != supported or set(matched) != supported
            or len(matched) != len(supported) or queries.keys() != supported or cohort['unavailable']):
        raise ValueError('Final audit coverage differs from the frozen human judgments.')
    for qid, row in queries.items():
        if row['support_alternatives'] != confirmed[qid]['support_sets'] or row['models'].keys() != set(MODELS):
            raise ValueError(f'Final audit evidence or model coverage differs for {qid}.')

    print('# SupportQA v1 human-label evaluation')
    print(f'\n{len(confirmed)} confirmed judgments, {len(supported)} with sufficient support, '
          f'and all {len(matched)} jointly scored by all four models.')
    tables = [(label, 'human', {q: queries[q]['models'][model]['metrics'] for q in matched}, matched)
              for model, label in zip(MODELS, ('APE', 'BGE-small', 'SimCSE', 'Contriever'), strict=True)]
    metric_table('Human-label retrieval results', tables, len(confirmed))


def report(study, root):
    audit_path = root / 'final-audit' / 'audit.json'
    if audit_path.is_file():
        return final_report(study, root, audit_path)

    print('# SupportQA local human evaluation')
    print('\nOnly explicitly confirmed versions enter metrics. Evaluation uses the latest saved decision '
          'before seeing model comparison. Diagnostic summaries use the latest saved decision and are descriptive. '
          'A missing saved prediction or unusable alignment is unavailable. An unmatched proposition in a usable '
          'APE alignment stays a miss in the sufficient-set denominator.')
    for queue in ('evaluation', 'diagnostic', 'second'):
        selection = open_queues(root, queue)
        ids = selection['query_ids']
        records = judgment_records(study, root, queue, ids)
        progress(queue, ids, records)
        if queue == 'second':
            print('Independent judgments are kept separate. Use reconcile after both reviews are saved.')
            continue
        confirmed = latest_confirmations(records, blinded=queue == 'evaluation')
        common_ids = set()
        if confirmed:
            human_silver_changes(study, confirmed)
            common_ids = print_scores(study, confirmed)
        else:
            print('No confirmed judgments for this analysis.')
        if queue == 'evaluation':
            population(study, selection, confirmed, common_ids)
        else:
            diagnoses = [r['diagnoses'][-1] for r in records.values() if r['diagnoses']]
            counts = Counter(cause for row in diagnoses for cause in row['causes'])
            print(f'{len(diagnoses)}/{len(ids)} cases have a diagnosis. Cause counts can overlap.')
            for cause, count in sorted(counts.items()):
                print(f'- {cause}: {count}')
            print('Targeted diagnostic cases do not contribute to population estimates.')


def reconcile(study, root):
    ids = open_queues(root, 'second')['query_ids']
    primary = judgment_records(study, root, 'evaluation', open_queues(root, 'evaluation')['query_ids'])
    second = judgment_records(study, root, 'second', ids)
    if any(q not in primary or q not in second or not current_version(primary[q]) or not current_version(second[q])
           for q in ids):
        raise ValueError('Both reviewers must save a decision on all 40 shared queries before reconciliation.')
    a, b = (latest_confirmations(rows, blinded=True) for rows in (primary, second))
    paired = [q for q in ids if q in a and q in b]
    print(f'Paired blinded confirmations: {len(paired)}/{len(ids)}. Decisions remain in separate files.')
    for qid in ids:
        if qid not in paired:
            print(f'{qid} pending blinded confirmation or adjudication')
        elif (a[qid]['source']['answerability'] != b[qid]['source']['answerability']
              or support_ids(a[qid]['support_sets']) != support_ids(b[qid]['support_sets'])):
            print(f'{qid} primary {a[qid]["source"]["answerability"]} {a[qid]["support_sets"]} / '
                  f'independent {b[qid]["source"]["answerability"]} {b[qid]["support_sets"]}')
