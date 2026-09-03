"""TPPS prototype demo: Plan A induction + Liouville render + commit gate."""
from tpps.core import typecheck as T
from tpps.domains import complex_analysis as CA
from tpps.domains import rewriting as RW
from tpps.core.evidence import Evidence, Reviewed
from tpps.examples import (complex_registry, example_registry, plan_a, plan_b,
                           plan_c_failure, theorem6_completion)
from tpps.adapters.render import Register, render
from tpps.config import load_config, make_store


def main():
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
