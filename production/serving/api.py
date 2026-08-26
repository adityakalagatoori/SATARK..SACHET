"""
Production HTTP API — authenticated, role-gated, backed by the governance
layer built in Phase 1.

Why this exists: dashboard/investigator.html (the investigator-facing
console) calls /api/alerts, /api/metrics, /api/stats — none of which exist
anywhere in src/satark.py's serve_api() (which only has /predict and
/health). This is that missing layer, built for real this time, with
authentication and RBAC (see auth.py) instead of open unauthenticated
routes.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'governance'))
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request
import kill_switch
import human_override
import explainability_store
from auth import require_permission, AuthorizationError
import io
import pandas as pd
from score_new_dataset import score_dataset, score_and_persist, load_upload, SchemaMismatchError, _load_uploads_index

app = Flask(__name__)


@app.after_request
def _allow_cors(resp):
    # The live-scoring / investigator dashboard pages are served as plain
    # static files (python -m http.server), a different origin than this API
    # process -- allow that cross-origin fetch/POST for the demo endpoints.
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return resp


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'HEALTHY',
        'kill_switch_active': kill_switch.is_active(),
    })


@app.route('/api/kill-switch/status', methods=['GET'])
@require_permission('view_kill_switch_history')
def kill_switch_status(caller):
    return jsonify(kill_switch.get_status())


@app.route('/api/kill-switch/history', methods=['GET'])
@require_permission('view_kill_switch_history')
def kill_switch_history(caller):
    return jsonify(kill_switch.get_history(limit=int(request.args.get('limit', 100))))


@app.route('/api/kill-switch/activate', methods=['POST'])
@require_permission('kill_switch_activate')
def kill_switch_activate(caller):
    reason = (request.json or {}).get('reason')
    if not reason:
        return jsonify({'error': 'reason is required'}), 400
    kill_switch.activate(triggered_by=caller['user_id'], reason=reason)
    return jsonify({'status': 'ACTIVATED', 'by': caller['user_id']})


@app.route('/api/kill-switch/deactivate', methods=['POST'])
@require_permission('kill_switch_deactivate')
def kill_switch_deactivate(caller):
    reason = (request.json or {}).get('reason')
    if not reason:
        return jsonify({'error': 'reason is required'}), 400
    kill_switch.deactivate(triggered_by=caller['user_id'], reason=reason)
    return jsonify({'status': 'DEACTIVATED', 'by': caller['user_id']})


@app.route('/api/accounts/<int:account_id>/actions', methods=['GET'])
@require_permission('view_alerts')
def account_actions(caller, account_id):
    return jsonify(human_override.get_action_status(account_id))


@app.route('/api/accounts/<int:account_id>/override', methods=['POST'])
@require_permission('override_action')
def override_account_action(caller, account_id):
    body = request.json or {}
    action_id = body.get('action_id')
    reason = body.get('reason')
    if not action_id or not reason:
        return jsonify({'error': 'action_id and reason are required'}), 400
    human_override.override_action(action_id, investigator_id=caller['user_id'], reason=reason)
    return jsonify({'status': 'OVERRIDDEN', 'action_id': action_id, 'by': caller['user_id']})


@app.route('/api/accounts/<int:account_id>/explanation', methods=['GET'])
@require_permission('view_explanation')
def account_explanation(caller, account_id):
    explanation = explainability_store.get_explanation(account_id)
    if explanation is None:
        return jsonify({'error': 'no explanation stored for this account'}), 404
    return jsonify(explanation)


@app.route('/api/actions/active', methods=['GET'])
@require_permission('view_alerts')
def active_actions(caller):
    return jsonify(human_override.list_active_actions(action_type=request.args.get('action_type')))


@app.route('/api/score-dataset', methods=['POST', 'OPTIONS'])
def score_dataset_endpoint():
    """Score an uploaded CSV (same raw F-column schema as data/DataSet (1).csv)
    through the FULL bundled ensemble (see production/serving/score_new_dataset.py
    for exactly which artifacts are applied and which had to be reconstructed
    from the original training data because they were never persisted), and
    persist the result under its own upload id so it can be selected later in
    the Investigator Console / opened in the 3D view -- it does NOT overwrite
    any previous upload.

    Deliberately NOT behind @require_permission: this is the live-scoring demo
    entry point (dashboard/live_scoring.html), which has no login flow, and it
    only ever operates on data the caller themselves just uploaded -- unlike
    every other route in this file, which touches persistent account/alert
    state and stays permission-gated.

    Request:  multipart/form-data, field name 'file', a .csv upload of any size.
    Response: {"upload_id": "...", "summary": {...}} on success. If the
              uploaded file has no F3924 ground-truth column, summary.metrics
              is null rather than a fabricated number.
    On a schema mismatch (uploaded columns don't match the trained schema),
    returns 422 with the exact column diff instead of silently scoring anyway.
    """
    if request.method == 'OPTIONS':
        return ('', 204)
    if 'file' not in request.files:
        return jsonify({'error': "no file uploaded -- expected multipart field 'file'"}), 400
    f = request.files['file']
    try:
        raw_df = pd.read_csv(io.BytesIO(f.read()))
    except Exception as e:
        return jsonify({'error': f'could not parse uploaded file as CSV: {e}'}), 400

    compute_shap = request.args.get('shap', 'true').lower() != 'false'

    try:
        result = score_and_persist(raw_df, filename=f.filename or 'upload.csv', compute_shap=compute_shap)
    except SchemaMismatchError as e:
        return jsonify({'error': 'schema_mismatch', 'detail': str(e)}), 422
    except FileNotFoundError as e:
        return jsonify({'error': 'server_artifact_missing', 'detail': str(e)}), 500
    except Exception as e:
        return jsonify({'error': 'scoring_failed', 'detail': str(e)}), 500

    return jsonify(result)


@app.route('/api/uploads', methods=['GET'])
def list_uploads():
    """List every previously scored upload (id, filename, time, counts) --
    the Investigator Console's dataset selector reads this to populate its
    dropdown alongside the original competition dataset."""
    return jsonify(_load_uploads_index())


@app.route('/api/uploads/<upload_id>', methods=['GET'])
def get_upload(upload_id):
    """Full scored result for one upload (meta + every account, including
    x/y/z universe coordinates and SHAP) -- same shape score_dataset()
    returns. Used by the Investigator Console (table/tiers) and by
    account_detail_3d.html?dataset=<upload_id> for a single account's detail."""
    result = load_upload(upload_id)
    if result is None:
        return jsonify({'error': f'no upload found with id {upload_id!r}'}), 404
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
