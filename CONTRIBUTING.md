# Contributing to projectaria_hloc_example
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes (`python -m pytest`).
5. If you haven't already, complete the Contributor License Agreement ("CLA").

## Tests and GitHub CI

The GitHub workflow runs on pushes to `main` (including ShipIt syncs), pull
requests targeting `main`, and manual dispatches. It runs the mocked core suite
on Ubuntu and macOS, plus the native COLMAP pose tests on Ubuntu. These cover
central, two-camera and four-camera geometry without glasses, recordings, online
MPS, or model weights.

For the lightweight core suite in Python 3.12:

```bash
python -m pip install --only-binary=:all: -r .github/ci-requirements.txt
python -m pytest tests --ignore=tests/test_native_pose.py
```

To run the native geometry tests as well:

```bash
python -m pip install --only-binary=:all: pycolmap==4.2.1
python -m pytest tests/test_native_pose.py
```

This is regression coverage, not an end-to-end recording or live-device
qualification. Workflow runs use read-only repository permissions and hosted
runners, with no repository secrets. Changes to the workflow follow the same
review path as source changes; do not add GitHub-only files to synced `main`.

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
* 4 spaces for indentation rather than tabs
* Match the style of the surrounding code.

## License
By contributing to projectaria_hloc_example, you agree that your
contributions will be licensed under the LICENSE file in the root directory of
this source tree.
