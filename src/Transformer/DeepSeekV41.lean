module

public import Transformer.DeepSeekV41.Section2_SinglePass

/-!
# DeepSeek V4.1 Single-Pass mHC

DeepSeek-V4.1-Flash technical report, §2.4.1, Eqs. (2)–(6), downloaded
to `papers/DeepSeek-V4.1-Flash/DeepSeek_V41_Tech_Report.pdf` from the
official checkpoint. The read map is delayed by one sublayer; the current
skip and write maps retain their dependence on the current streams.
The model records the initial read map explicitly and proves that the
shift can change the output. The activation traffic count is halved under
the report's stated cost model. Kernel coefficients, including transpose
and epsilon, are linked to the separate finite implementation model.

Only this residual section is formalized. Other architectural changes
and empirical claims are outside the scope. See `review/residual_connections.md`.
-/
