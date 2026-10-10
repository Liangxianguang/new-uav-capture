"""Mock-only Git transport safety, never remote publication evidence."""
import hashlib
import importlib.util
import types
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('exact_publisher', Path(__file__).resolve().parents[1]/'scripts/publish_exact_github_api.py')
pub = importlib.util.module_from_spec(spec); spec.loader.exec_module(pub)


def test_identity_preserves_timezone_and_terminal_message():
    result = pub.identity(b'Example User <user@example.invalid> 1791662149 +0800')
    assert result['date'].endswith('+08:00')
    raw = (b'tree '+b'a'*40+b'\nparent '+b'b'*40+
        b'\nauthor Example <user@example.invalid> 1 +0000\ncommitter Example <user@example.invalid> 2 -0530\n\nmessage\n')
    result = pub.commit_payload(raw)
    assert result['message'] == 'message\n'
    assert result['committer']['date'].endswith('-05:30')


@pytest.mark.parametrize('header', [b'gpgsig signature', b'encoding utf-8', b'parent '+b'c'*40])
def test_unsupported_signed_or_nonstandard_commits_refused(header):
    raw = (b'tree '+b'a'*40+b'\nparent '+b'b'*40+
        b'\nauthor Example <user@example.invalid> 1 +0000\ncommitter Example <user@example.invalid> 2 +0000\n'+header+b'\n\nmessage\n')
    with pytest.raises(ValueError): pub.commit_payload(raw)


@pytest.fixture
def objects():
    base, target, old_tree, tree = 'a'*40, 'b'*40, 'c'*40, 'd'*40
    def commit(t, p):
        return (f'tree {t}\nparent {p}\nauthor Example <user@example.invalid> 1 +0000\n'
            'committer Example <user@example.invalid> 2 +0000\n\nmessage\n').encode()
    blob = b'not a secret\n'; blob_sha = hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
    data = {('commit', base): commit(old_tree, 'e'*40), ('commit', target): commit(tree, base), ('blob', blob_sha): blob}
    return base, target, tree, blob_sha, data


class FakeAPI:
    def __init__(self, base, target, tree, blob, drift=None):
        self.remote = base; self.target = target; self.tree = tree; self.blob = blob
        self.drift = drift; self.calls = []; self.reads = 0

    def head(self):
        self.reads += 1
        return 'f'*40 if self.drift == 'concurrent' and self.reads >= 2 else self.remote

    def request(self, method, path, payload):
        self.calls.append((method, path, payload))
        if path == '/git/blobs': result = {'sha': self.blob}; kind = 'blob'
        elif path == '/git/trees': result = {'sha': self.tree}; kind = 'tree'
        elif path == '/git/commits': result = {'sha': self.target}; kind = 'commit'
        else:
            assert method == 'PATCH' and payload['force'] is False
            self.remote = payload['sha']
            return {'ref': 'refs/heads/'+pub.BRANCH, 'object': {'sha': self.remote}}
        if self.drift == kind: result['sha'] = 'f'*40
        return result


def test_exact_objects_only_then_nonforce_final_ref(objects):
    base, target, tree, blob, data = objects; api = FakeAPI(base, target, tree, blob)
    result = pub.publish_objects(api, [target], base, target, lambda k, s: data[k, s], lambda p, s: [('docs/report.md', '100644', blob)])
    assert result['remote_head'] == target and result['commits_rewritten'] is False
    assert [method for method, _, _ in api.calls] == ['POST', 'POST', 'POST', 'PATCH']
    assert api.calls[-1][2] == {'sha': target, 'force': False}


@pytest.mark.parametrize('drift', ['blob', 'tree', 'commit', 'concurrent'])
def test_any_remote_object_or_concurrent_drift_prevents_ref_update(objects, drift):
    base, target, tree, blob, data = objects; api = FakeAPI(base, target, tree, blob, drift)
    with pytest.raises(ValueError):
        pub.publish_objects(api, [target], base, target, lambda k, s: data[k, s], lambda p, s: [('docs/report.md', '100644', blob)])
    assert not any(method == 'PATCH' for method, _, _ in api.calls)
    assert api.remote == base


@pytest.mark.parametrize('path,mode', [('../outside', '100644'), ('/outside', '100644'), ('a\\b', '100644'), ('a', '160000')])
def test_path_and_mode_scope_refused_before_blob_upload(objects, path, mode):
    base, target, tree, blob, data = objects; api = FakeAPI(base, target, tree, blob)
    with pytest.raises(ValueError):
        pub.publish_objects(api, [target], base, target, lambda k, s: data[k, s], lambda p, s: [(path, mode, blob)])
    assert not api.calls


def test_stale_expected_remote_prevents_all_writes(objects):
    base, target, tree, blob, data = objects; api = FakeAPI('f'*40, target, tree, blob)
    with pytest.raises(ValueError): pub.publish_objects(api, [target], base, target, lambda k, s: data[k, s], lambda p, s: [])
    assert not api.calls


def test_existing_credential_is_captured_not_logged(monkeypatch, capsys):
    def fake_run(*args, **kwargs):
        assert kwargs['capture_output'] is True
        assert kwargs['env']['GIT_TERMINAL_PROMPT'] == '0'
        assert kwargs['env']['GCM_INTERACTIVE'] == 'never'
        return types.SimpleNamespace(returncode=0, stdout=b'username=fixture\npassword=SYNTHETIC_SECRET\n', stderr=b'')
    monkeypatch.setattr(pub.subprocess, 'run', fake_run)
    assert pub.credential_token() == 'SYNTHETIC_SECRET'
    assert capsys.readouterr().out == ''


def test_failed_credential_does_not_expose_captured_streams(monkeypatch, capsys):
    monkeypatch.setattr(pub.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(returncode=1,
        stdout=b'SYNTHETIC_SECRET', stderr=b'SYNTHETIC_SECRET'))
    with pytest.raises(ValueError) as error: pub.credential_token()
    assert 'SYNTHETIC_SECRET' not in str(error.value)
    assert capsys.readouterr().out == ''


def test_redirects_are_refused_before_forwarding_credentials():
    with pytest.raises(ValueError): pub.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.invalid')
