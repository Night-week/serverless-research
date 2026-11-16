# Rolling-transition helper

`rolling_transition.py` contains the version-overlap transition semantics used by the R218 evaluator. It reuses the shared event engine and falls back to global drain if the old and new package versions cannot coexist within the modeled RAM capacity.

The separate R211 response-delay sensitivity runner and its delay variants are intentionally not included. R218 calls only `Rolling` and `local` from this module and supplies its single frozen decision-latency estimate as an argument.
