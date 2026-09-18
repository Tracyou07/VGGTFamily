# Publication verification — 2026-09-18

CPU-only verification on H20 before publication:

- ScanNet: `python -m unittest discover -s tests -q` in vggt-gx: 52 tests, OK, 1 skipped.
- KITTI: `PYTHON=/home/ubuntu/anaconda3/envs/fastwam/bin/python bash scripts/verify_all_cpu.sh`: 301 passed. Script also checked CLI help, shell syntax, and unchanged source state.
- Virtual KITTI: same CPU verification script in its directory: 214 passed; CLI help, shell syntax, and unchanged source state checked.
- Published snapshot: AST parsing passed for 146 Python files; all bundled JSON files parsed.
- Common credential/key patterns and prohibited artifact filenames were checked before staging. No datasets, weights, runtime dependencies, or results were included.

No new real-model, GPU, or dataset benchmark was launched. 7-Scenes did not receive a new full test run during publication. Deployment-specific paths remain documented and require adjustment on other machines. Existing whitespace in upstream/reference code and license text was preserved.