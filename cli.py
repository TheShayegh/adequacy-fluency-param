"""Single entry point mapping every paper figure/table to a command.

Each subcommand forwards straight to the matching mwb/scripts/*.py module
(run as `python -m mwb.scripts.<name>`, so its own imports resolve with no
path hacks) -- this file adds no computation of its own, only discoverability
and a stable name per paper artifact. Every script also still runs standalone
(`python -m mwb.scripts.compute_dataset_stats --help`) with its own, more
detailed --help; use that for the full flag reference.

Usage: python cli.py <command> [-- SCRIPT_ARGS...]
       python cli.py <command> --help          (this file's one-line summary)
       python cli.py <command> -- --help        (the underlying script's own --help)

Examples:
  python cli.py dataset-stats
  python cli.py af-scatter
  python cli.py preference-compute -- --dataset ende21
  python cli.py preference-plot -- --dataset ende21 --poster --format pdf
  python cli.py beta-ess-heatmap -- --dataset ende21 --grid-n 32
  python cli.py solver-efficiency -- --dataset ende24
"""

import argparse
import subprocess
import sys

# name -> (module, one-line description, paper artifact)
COMMANDS = {
    'dataset-stats': (
        'compute_dataset_stats',
        'Per-dataset descriptive statistics.',
        'Appendix "Dataset Statistics" (Tables dataset-core/betaij/mqm/scorers)'),
    'af-scatter': (
        'plot_af_scatter_heen23_jazh24',
        'Adequacy-vs-fluency scatter for heen23/jazh24.',
        'Figure 1'),
    'spa-plane-synthetic': (
        'plot_spa_plane_synthetic_af_combo',
        'Synthetic scorer families on the SPA plane.',
        'Figure 2'),
    'preference-compute': (
        'compute_scorer_preference_vs_beta',
        'Computes preference curves vs. beta for one dataset (run this first).',
        'Figure 3 data (main results)'),
    'preference-synth25': (
        'compute_synth25_preference',
        "Computes Shayegh et al. (2025)'s system-synthesis baseline markers.",
        'Figure 3 markers (Shayegh et al. 2025 baseline)'),
    'preference-plot': (
        'plot_scorer_preference_vs_beta',
        'Plots the cached preference curves (run preference-compute first).',
        'Figure 3 and its appendix grid'),
    'beta-ess-heatmap': (
        'compute_beta_ess_preference_heatmap',
        'Adequacy-over-fluency preference over the (ESS, beta) plane.',
        'Figure 4'),
    'validate-beta-ess': (
        'validate_beta_ess_preference',
        'Cross-checks the beta-ESS heatmap sampler against the exhaustive lattice.',
        'validation for Figure 4'),
    'spa-vs-beta-compute': (
        'compute_spa_vs_beta',
        'Computes weighted-SPA(beta) curves for a set of scorers.',
        'Figure 5 data'),
    'spa-vs-beta-plot': (
        'plot_spa_vs_beta_combo',
        'Plots the cached weighted-SPA(beta) curves.',
        'Figure 5'),
    'loo-tau': (
        'compute_loo_tau',
        'Pooled Kendall tau across leave-p-out subsets, per dataset.',
        'Table dataset_tau (Appendix, LOO stability)'),
    'loo-tau-vs-beta-compute': (
        'compute_loo_tau_vs_beta',
        'Computes pooled LOO tau swept over target beta, for one dataset.',
        'Figure 6 data'),
    'loo-tau-vs-beta-plot': (
        'plot_loo_tau_vs_beta_combo',
        'Plots the cached LOO-tau-vs-beta curves.',
        'Figure 6'),
    'solver-efficiency': (
        'solver_efficiency',
        'Ablates the three pruning theorems and counts FLOPs/supports/candidates.',
        'Tables solve-exact-operation-counts / solve-exact-theorem-invocations'),
    'validate-scores': (
        'validate_scores',
        'Checks reconstructed MQM scores against the official WMT score files.',
        'data-pipeline validation, not a paper artifact'),
}


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  subparsers = parser.add_subparsers(dest='command', metavar='<command>')
  for name, (module, desc, artifact) in sorted(COMMANDS.items()):
    sub = subparsers.add_parser(name, help=f'{desc} -> {artifact}')
    sub.add_argument('script_args', nargs=argparse.REMAINDER,
                      help="arguments forwarded to the script, e.g. -- --dataset ende21")

  args = parser.parse_args(argv)
  if args.command is None:
    parser.print_help()
    return 1

  module, _, _ = COMMANDS[args.command]
  script_args = args.script_args
  if script_args and script_args[0] == '--':
    script_args = script_args[1:]
  cmd = [sys.executable, '-m', f'mwb.scripts.{module}', *script_args]
  return subprocess.run(cmd).returncode


if __name__ == '__main__':
  sys.exit(main())
