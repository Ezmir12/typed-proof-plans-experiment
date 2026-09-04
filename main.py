"""TPPS prototype demo: Plan A induction + Liouville render + commit gate."""
import argparse

from tpps.core import typecheck as T
from tpps.domains import complex_analysis as CA
from tpps.domains import rewriting as RW
from tpps.core.evidence import Evidence, Reviewed
from tpps.examples import (complex_registry, example_registry, plan_a, plan_b,
                           plan_c_failure, theorem6_completion)
from tpps.adapters.render import Register, render
from tpps.config import load_config, make_store


def main():
    ap = argparse.ArgumentParser(description="TPPS prototype demo / NL pipeline.")
    ap.add_argument("--pipeline", metavar="TEXT", default=None,
                    help="run one NL string through the logged pipeline "
                         "(extractor: TPPS_EXTRACTOR=llm|demo|hooks, default llm)")
    ap.add_argument("--extractor", choices=("llm", "demo"), default=None,
                    help="override the configured extractor for this run")
    ap.add_argument("--log-dir", default="logs/runs")
    args = ap.parse_args()
    if args.pipeline is not None:
        from tpps.pipeline import run_pipeline
        res = run_pipeline(args.pipeline, log_dir=args.log_dir,
                           extractor=args.extractor)
        print(f"run_id: {res.run_id}\nlog: {res.log_path}\n"
              f"accepted: {res.accepted} target_closed: {res.target_closed}")
        print(f"replay: uv run python tools/replay_log.py {res.run_id} "
              f"--log-dir {args.log_dir}")
        return
    settings = load_config()
    print(f"== config: llm={settings.llm.provider}/{settings.llm.model} "
          f"kernel={settings.kernel.env_hash} store={settings.store.owner}")
    reg = example_registry()
    print("== Plan A check:", T.check_plan(plan_a(), reg).ok)
    print("== Plan B check:", T.check_plan(plan_b(), reg).ok)
    ind = theorem6_completion()
    print("== Induced:", ind.provenance)
    print("== Plan C:", plan_c_failure().critic_input())

    liouv = CA.liouville_plan()
    req = Evidence(reviewed=Reviewed.INDEPENDENT)
    print("--- Liouville (textbook) ---")
    print(render(liouv, "constant", req, Register.TEXTBOOK))
    print("--- Liouville (annotated) ---")
    print(render(liouv, "constant", req, Register.ANNOTATED))

    store = make_store(settings)
    rec = store.commit_gate_typing_pass(
        plan_a().to_atoms(),
        lambda: ((r := T.check_plan(plan_a(), reg)).ok, r.diagnostic))
    print("== commit gate:", rec.check_ok, rec.content_hash)
    print("== rewriting schemas:", len(RW.schema_library().schemas),
          "| complex schemas:", len(CA.schema_library().schemas))


if __name__ == "__main__":
    main()
