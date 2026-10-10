"""Fallback transport for this research branch, preserving exact Git objects.

Uses existing Git credentials only. Never prints/stores credentials, rewrites
commits, changes Git configuration, or force-updates a ref. Default is read-only.
Only standard unsigned single-parent commits and regular files are supported.
Every remote blob/tree/commit SHA must equal the local object before ref update.
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = 'Liangxianguang/new-uav-capture'
BRANCH = 'causal-world-model-v1-20261009'
API_ROOT = 'https://api.github.com/repos/'+REPOSITORY
ROOT = Path(__file__).resolve().parents[1]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise ValueError('Refuse API redirect; never forward credential to another endpoint')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def identity(value):
    match = re.fullmatch(rb'(.*) <([^<>\r\n]+)> (\d+) ([+-]\d{4})', value)
    if not match:
        raise ValueError('Unsupported Git identity')
    name, email, seconds, zone = match.groups()
    offset = int(zone[1:3])*60+int(zone[3:])
    if zone[:1] == b'-': offset = -offset
    moment = datetime.datetime.fromtimestamp(int(seconds),
        tz=datetime.timezone(datetime.timedelta(minutes=offset)))
    return {'name': name.decode('utf8'), 'email': email.decode('utf8'), 'date': moment.isoformat()}


def commit_payload(raw):
    headers, message = raw.split(b'\n\n', 1)
    fields = {}
    for line in headers.splitlines():
        key, value = line.split(b' ', 1)
        if key not in (b'tree', b'parent', b'author', b'committer') or key in fields:
            raise ValueError('Only standard unsigned linear Git commits are supported')
        fields[key] = value
    if set(fields) != {b'tree', b'parent', b'author', b'committer'}:
        raise ValueError('Complete single-parent commit required')
    return {'tree': fields[b'tree'].decode('ascii'), 'parents': [fields[b'parent'].decode('ascii')],
        'author': identity(fields[b'author']), 'committer': identity(fields[b'committer']),
        'message': message.decode('utf8')}


def credential_token():
    task_env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
    result = subprocess.run(['git', '-c', 'credential.interactive=never', 'credential', 'fill'],
        input=b'protocol=https\nhost=github.com\n\n', cwd=ROOT, capture_output=True,
        env=task_env, timeout=30)
    # Never propagate credential-manager stdout/stderr or the token into errors.
    if result.returncode:
        raise ValueError('Existing noninteractive Git credential unavailable')
    values = dict(line.split(b'=', 1) for line in result.stdout.splitlines() if b'=' in line)
    token = values.get(b'password')
    if not token:
        raise ValueError('Existing Git credential lacks a password/token')
    return token.decode('utf8')


class GitHubAPI:
    def __init__(self, token=None): self.token = token

    def request(self, method, suffix, payload=None):
        if not suffix.startswith('/git/'):
            raise ValueError('Only repository-scoped Git-data API is allowed')
        headers = {'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'uav-cwm-exact-object-publication', 'Content-Type': 'application/json',
            'Cache-Control': 'no-cache'}
        if self.token: headers['Authorization'] = 'Bearer '+self.token
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf8')
        request = urllib.request.Request(API_ROOT+suffix, data=data, headers=headers, method=method)
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise ValueError(f'GitHub Git-data API HTTP status {error.code}') from None
        except urllib.error.URLError:
            raise ValueError('GitHub Git-data API transport unavailable') from None

    def head(self):
        result = self.request('GET', '/git/ref/heads/'+BRANCH)
        if result.get('ref') != 'refs/heads/'+BRANCH or result['object']['type'] != 'commit':
            raise ValueError('Exact repository branch identity required')
        return result['object']['sha']


def verify_sha(result, expected, kind):
    if result.get('sha') != expected:
        raise ValueError(f'Exact {kind} object SHA differs; refuse branch update')


def publish_objects(api, commits, expected_remote, target, object_reader, change_reader):
    """Injectable exact-object publisher; branch is never force-updated."""
    if api.head() != expected_remote:
        raise ValueError('Remote branch changed; refuse stale publication')
    parent = expected_remote
    for sha in commits:
        payload = commit_payload(object_reader('commit', sha))
        if payload['parents'] != [parent]:
            raise ValueError('Publication must be a complete linear fast-forward')
        base_tree = commit_payload(object_reader('commit', parent))['tree']
        tree = []
        for path, mode, blob_sha in change_reader(parent, sha):
            if (path.startswith('/') or '\\' in path or any(part in ('', '.', '..') for part in path.split('/'))
                or mode != '100644' or not re.fullmatch(r'[0-9a-f]{40}', blob_sha)):
                raise ValueError('Only exact regular repository file updates are supported')
            raw = object_reader('blob', blob_sha)
            local_sha = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if local_sha != blob_sha:
                raise ValueError('Local blob reader identity differs')
            blob = api.request('POST', '/git/blobs', {'content': base64.b64encode(raw).decode('ascii'), 'encoding': 'base64'})
            verify_sha(blob, blob_sha, 'blob')
            tree.append({'path': path, 'mode': mode, 'type': 'blob', 'sha': blob_sha})
        created_tree = api.request('POST', '/git/trees', {'base_tree': base_tree, 'tree': tree})
        verify_sha(created_tree, payload['tree'], 'tree')
        created_commit = api.request('POST', '/git/commits', payload)
        verify_sha(created_commit, sha, 'commit')
        parent = sha
        print(json.dumps({'exact_remote_commit_object_verified': sha}), flush=True)
    if parent != target or api.head() != expected_remote:
        raise ValueError('Target mismatch or concurrent branch movement; refuse ref update')
    if commits:
        updated = api.request('PATCH', '/git/refs/heads/'+BRANCH, {'sha': target, 'force': False})
        if updated.get('ref') != 'refs/heads/'+BRANCH or updated['object']['sha'] != target:
            raise ValueError('Fast-forward ref update result differs; inspect before retry')
    if api.head() != target:
        raise ValueError('Exact final remote branch confirmation failed; inspect before retry')
    return {'status': 'exact_existing_git_commits_published_fast_forward', 'repository': REPOSITORY,
        'branch': BRANCH, 'previous_remote_head': expected_remote, 'remote_head': target,
        'commits': commits, 'all_object_shas_equal': True, 'force_update': False,
        'commits_rewritten': False, 'git_configuration_changed': False}


def changes(parent, commit):
    entries = []
    paths = git('diff-tree', '--no-commit-id', '--name-only', '-r', '-z', parent, commit).split(b'\0')
    for raw_path in filter(None, paths):
        path = raw_path.decode('utf8')
        record = git('ls-tree', '-z', commit, '--', path).split(b'\0')[0]
        meta, actual_path = record.split(b'\t', 1)
        mode, kind, sha = meta.split()
        if actual_path != raw_path or kind != b'blob':
            raise ValueError('Deletion/rename/submodule publication is unsupported')
        entries.append((path, mode.decode(), sha.decode()))
    return entries


def run(expected_remote, publish=False, authenticated_read=False):
    if not re.fullmatch(r'[0-9a-f]{40}', expected_remote):
        raise ValueError('Explicit exact expected remote commit required')
    if git('branch', '--show-current').decode().strip() != BRANCH:
        raise ValueError('Only the existing research branch is in scope')
    if git('diff', '--name-only', 'HEAD'):
        raise ValueError('Tracked changes must be committed before exact publication')
    target = git('rev-parse', 'HEAD').decode().strip()
    if subprocess.run(['git', 'merge-base', '--is-ancestor', expected_remote, target], cwd=ROOT).returncode:
        raise ValueError('Existing remote HEAD is not a local ancestor')
    commits = git('rev-list', '--reverse', expected_remote+'..'+target).decode().splitlines()
    if len(commits) > 20:
        raise ValueError('Refuse unexpectedly broad publication')
    api = GitHubAPI(credential_token() if publish or authenticated_read else None)
    if not publish:
        remote = api.head()
        if remote != expected_remote: raise ValueError('Remote expected base differs')
        return {'status': 'read_only_exact_publication_plan', 'remote_head': remote,
            'target': target, 'commits': commits, 'external_writes': False}
    return publish_objects(api, commits, expected_remote, target, lambda kind, sha: git('cat-file', kind, sha), changes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-remote-head', required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--authenticated-read', action='store_true',
        help='Use existing Git credential for read-only API verification')
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.expected_remote_head, args.publish, args.authenticated_read), indent=2))
    except Exception as error:
        # No traceback, request headers, credential streams or arbitrary payloads.
        if isinstance(error, ValueError): print(str(error))
        else: print('Exact publication failed; inspect remote state before retry')
        raise SystemExit(1)
