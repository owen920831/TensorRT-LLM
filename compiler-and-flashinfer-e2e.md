<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 從一個 Request 到 GPU：TensorRT-LLM、編譯器與 FlashInfer

離線視覺化入口：[e2e-learning.html](e2e-learning.html)。課程地圖與唯一完成清單：[interview-preparation.md](interview-preparation.md)。
學習回答與驗收：[learning-progress.md](learning-progress.md)。本文件是概念教材的單一編輯來源；HTML 是產生的閱讀版本。

## 00｜來源、範圍與閱讀方式

查閱日期：2026-10-27；TensorRT-LLM checkout：`e921fa39b07c669f2633ac95390fd9deac32d92b`。

你提供的是 Cishoon 的「torch 編譯器」合集，共 13 集、3 小時 29 分 16 秒。
已取得並交叉核對標題、BV ID、片長；完整清單與限制見 [影片來源紀錄](bilibili-compiler-sources.md)。
**未觀看完整影片、未取得逐字稿。** 影片頁與 view API 回傳 412；匿名 player API 成功，但 13 集均未回傳字幕清單。
這只代表目前匿名存取拿不到字幕，不能推斷影片沒有字幕。
以下逐集段落依影片標題安排主題，技術內容來自官方文件、本地原始碼與原創教學推導；不是講者逐字或逐段摘要。
FlashInfer 是依你的要求補充，不聲稱合集已涵蓋它。

本 checkout 的 [requirements.txt](requirements.txt) 指定 `flashinfer-python==0.6.18`、`triton==3.8.0`，
PyTorch 約束為 `>=2.14.0a0,<=2.14.0`。這是相依宣告，**不是本機已安裝版本的證據**。
外部 FlashInfer 文件查閱時顯示 0.7.2.post1；以它說明概念，實際 API 與支援範圍以 checkout 和對應版本為準。

推薦閱讀三輪：

1. 第一輪讀 01～03：知道每個名字在哪一層，以及一個 request 怎麼走完。
2. 第二輪讀 04～16：按 13 集主題理解編譯流程，每集對照一個 repo 接點。
3. 第三輪讀 17～23：深入 FlashInfer、Graph、KV metadata 與效能驗證，再做練習。

26～28 是回接完整模型的補充：attention 原理、兩筆 requests 的手算，以及多 GPU 的架構變化。

每節只抓「解決什麼問題、輸入輸出、架構位置、限制」。看過教材不代表已驗收。

## 01｜先把這些名字放回正確的層

| 層次 | 元件 | 負責的事 | 不可混淆的邊界 |
|---|---|---|---|
| Serving / runtime | TensorRT-LLM、vLLM、SGLang | 接請求、排程、資源管理、組 batch、執行模型、交付回應 | 是完整推論系統；不能只用一個 kernel 的速度比較 |
| 模型程式 | PyTorch modules | 描述 attention、MLP、normalization 等運算與資料關係 | 模型寫成 Python，不代表底下 GPU 運算由 Python 計算 |
| 圖擷取 | TorchDynamo | 分析 Python 執行、取得 FX 圖與重用條件 | 不是 CUDA Graph capture |
| 圖轉換 | AOTAutograd、decomposition、functionalization | 整理運算、處理 mutation；訓練時也處理 forward/backward 關係 | 名稱包含 autograd，不代表 inference 一定算 backward |
| Lowering / codegen | TorchInductor | 從圖與迴圈表示安排計算，產生 kernel 或呼叫外部實作 | 不保證所有算子都生成 Triton |
| GPU 程式開發 | CUDA C++、Triton、CuTe DSL | 寫出要在 GPU 執行的運算與資料配置 | Triton 與 CuTe DSL 不是 serving server |
| Kernel / operator libraries | cuBLAS、CUTLASS、FlashInfer、FlashAttention 實作 | 提供可重用或可組合的高效算子 | API 名稱不一定等於最終執行的 kernel 家族 |
| 提交 GPU 工作 | CUDA streams、CUDA Graphs | 定義順序、依賴，降低重複提交成本 | Graph replay 本身不等於把多個 kernel 融成一個 |
| 硬體 | SM、warp、register、shared memory、L2、HBM、Tensor Core | 實際執行與搬運資料 | 一個瓶頸可以來自多層，不能只看利用率百分比 |

FlashAttention 一詞常指 IO-aware exact attention 演算法家族，也可能指其程式庫實作。
FlashInfer 是提供 attention 等推論算子的 library。兩者不能只當作互斥品牌；某些 library 路徑會使用不同 attention kernel 家族。
[FlashInfer 官方介紹](https://github.com/flashinfer-ai/flashinfer)、[CUTLASS 官方介紹](https://github.com/NVIDIA/cutlass)。

## 02｜一個 Request 的完整旅程

以下用單 GPU、decoder-only、自回歸生成作為基準。多模態、MoE 與 speculative decoding 之後再加。

### 初始化：先把可重用的工作準備好

讀 model config 與權重，建立 PyTorch model，選算子與 attention backend，配置 workspace 和 KV 資源。
依實際選項執行 warmup、JIT、autotune、compile、Graph capture。這些成本可能發生在不同階段，不能全算成一次「編譯」。
入口：[model_loader.py](tensorrt_llm/_torch/pyexecutor/model_loader.py)、[model_engine.py](tensorrt_llm/_torch/pyexecutor/model_engine.py)。

### 接收：文字變成 Request

API 接收 prompt 與 sampling 設定，輸入處理得到 token IDs，request 保存生成狀態與限制。
Streaming 決定回應的交付方式，與模型每輪如何計算是相關但不同的介面。
入口：[BaseLLM.generate / generate_async](tensorrt_llm/llmapi/llm.py)、[LlmRequest](tensorrt_llm/_torch/pyexecutor/llm_request.py)。

### 選工作：Scheduler 決定這輪執行誰

檢查 request 是否能容納、token budget 與 KV 資源，再組成本輪 batch。
這個 batch 可能是 prefill、decode 或 mixed；request 數量、query token 數量、KV 長度不是同一個數字。
入口：[PyExecutor._schedule](tensorrt_llm/_torch/pyexecutor/py_executor.py)、[scheduler.py](tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py)。

### 做運算：Model Engine 執行一次 Forward

準備張量與 metadata，選 eager、compiled callable、Graph replay 或相容的分段路徑。
Transformer 各層進行 norm、QKV projection、position encoding、attention、output projection 與 MLP。
使用哪個 kernel 由模型、dtype、shape、GPU、quantization 與設定共同決定。
入口：[PyTorchModelEngine.forward](tensorrt_llm/_torch/pyexecutor/model_engine.py)、[modeling_llama.py](tensorrt_llm/_torch/models/modeling_llama.py)。

### 選 Token：Logits 變成生成結果

logits 經過必要的 penalties、temperature、篩選與 sampling；結果更新 request state。
未完成的 request 留待下一輪，完成或取消的 request 依實作流程釋放資源並交付回應。
入口：[sampler.py](tensorrt_llm/_torch/pyexecutor/sampler/sampler.py)、[sampler/ops/flashinfer.py](tensorrt_llm/_torch/pyexecutor/sampler/ops/flashinfer.py)。

**本節檢查：** 編譯器優化 model forward，為什麼不等於它接手了整個 request scheduler？

## 03｜三條流程同時存在

```text
服務流程：Request → Scheduler → KV resources → Model forward → Sampling → Response
                                        ↑ 每個 iteration 的交界
編譯流程：Python → Dynamo / guards → 圖轉換 → backend / Inductor → generated / external kernels
                                        ↓ 可供執行的 callable
執行流程：metadata 更新 → eager launch / CUDA Graph replay → GPU 完成 → 更新 request
```

這些是概念與責任分工，不保證每一條邊都是直接函式呼叫。
編譯不是每個 request 都從零做；Graph replay 仍需要有效資料、metadata 與資源生命週期。

| 都叫 scheduler / plan | 管理的對象 | 例子 |
|---|---|---|
| Serving scheduler | request 與 token budget | 誰進 batch、長 prompt 切多少 |
| Compiler scheduler | 圖節點、迴圈、資料依賴 | 哪些運算能融合、先後順序 |
| FlashInfer planning | attention 工作切分與相關 metadata | page tables、head dims、workspace |
| GPU hardware scheduling | blocks、warps、SM 執行資源 | 哪些 block/warp 可被排上硬體 |

理解第 07 集的「調度器」時，先確認它指編譯器，不要與 PyExecutor request scheduling 混在一起。

## 04｜第 01 集主題：為什麼要算子融合

**問題：** GPU 做很簡單的逐元素運算時，算術可能不是主要成本。

以 `square(relu(x))` 為例，eager 可能先產生 ReLU intermediate，再執行平方。
若兩次各讀寫全部 FP16 元素，理想化 tensor traffic 為每元素 8 bytes；融合成一次讀、一次寫則為 4 bytes。
這是 tensor 讀寫量模型，不是實測 HBM traffic，也不包含 allocator、cache 或其他成本。

融合可省 intermediate 的讀寫與部分 launch 成本，但可能增加 register pressure、編譯時間或限制其他重用。
小 workload 可能偏 launch-bound，大 workload 可能偏 bandwidth-bound；不能把標題的「99%」套用所有情境。

**Repo：** [fused_relu2_triton.py](tensorrt_llm/_torch/fused_relu2_triton.py) 第 26 行的 `_relu2_kernel`。
這是手寫 Triton 融合案例，不是已證明由 Inductor 自動產生的輸出。

**練習：** 畫 eager 與 fused 版本每次 load/store；列出 dtype、NaN 與 aliasing 需要核對的條件。

## 05｜第 02 集主題：Dynamo 如何抓圖

**問題：** 編譯器如何從動態 Python 程式看見可優化的 tensor 運算？

Dynamo 透過 Python frame/bytecode 層的分析，將可處理的 PyTorch 操作整理成 FX graph，交給可替換 backend。
FakeTensor 可以攜帶 shape、dtype、stride 等資訊，而不執行對應的大型真實 tensor 計算。
不代表原始 Python 的任何副作用都不執行，也不代表第一次呼叫完全沒有真實運算。

無法納入的部分可能產生 graph break；在要求 fullgraph 的情況，這可能直接成為錯誤，而不是默默切圖。
資料值控制的 Python branching 與只依 shape 決定的分支，也要分開分析。
[Dynamo 官方說明](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamo_overview.html)。

**Repo：** [model_engine.py](tensorrt_llm/_torch/pyexecutor/model_engine.py) 第 490 行建立自訂 `Backend`，511 行附近傳給 `torch.compile`。
這是條件式路徑，要先讀第 434 行附近的 compile 設定與 model 分支。

**練習：** 用紙筆畫 `relu(x + y)` 的 FX 概念圖，區分 tensor metadata 與 tensor 內容。

## 06｜第 03 集主題：Guards 與動態形狀

**問題：** 同一份 compiled code 何時可以安全重用？

Guard 是編譯假設的檢查，例如 dtype、shape/stride、Python 值或 module state。
有符合的已編譯變體就重用；沒有時可能觸發 tracing/compilation，達到限制後的行為取決於版本與設定。
Graph break 是圖擷取的邊界；guard failure 是某份已編譯變體的假設不成立，兩者不是同義詞。

符號維度可讓一份程式支援多種尺寸，但不會消除所有假設與分支 specialization。
因此「換十種形狀只編譯兩次」要有實際程式與 logs 才能重現，不能視為固定規則。
[Guards 與 runtime overhead](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.reducing_guard_overhead.html)。

**Repo：** [Backend.__call__](tensorrt_llm/_torch/compilation/backend.py) 第 187 行，讀取 `FakeTensor` 中的 token 維度以銜接 piecewise 邏輯。
TRT-LLM compile graph 可支援哪些 shape，與它 capture 了哪些 CUDA Graph buckets，是兩個問題。

**練習：** 預測只改內容、改 shape、改 dtype、改 stride，各可能使哪種假設失效；再用 logs 驗證。

## 07｜第 04 集主題：AOTAutograd 為什麼出現在這裡

**問題：** 訓練時 forward/backward 如何一起被整理？推論又為何見到這個名字？

訓練需要 backward，編譯流程可分析 forward 保存哪些值、哪些值重算，權衡運算與記憶體。
這是 compiler 的圖轉換階段，不代表 Python model 需要改成另一種訓練框架。

推論沒有梯度需求時，不應因此虛構 backward 計算；但框架仍可能重用這一層的圖整理與 decomposition infrastructure。
**本 checkout 的直接證據：** [backend.py](tensorrt_llm/_torch/compilation/backend.py) 第 222 行呼叫
`aot_module_simplified(..., fw_compiler=self.optimize, decompositions=select_decomp_table())`。

**練習：** 沿這個呼叫找 `fw_compiler` 實際是哪個函式；說明為何看到 `aot_autograd` import 不足以證明 inference 執行 backward。

## 08｜第 05 集主題：Decomposition 與 Functionalization

**問題：** 編譯器如何支援大量高階算子，以及有原地修改的程式？

Decomposition 把選定高階操作轉成較基礎的操作組合，減少後端需要獨立支援的語意。
Functionalization 用較容易分析的形式表達 mutation 與 aliasing，讓資料相依關係清楚。
這不是允許忽略副作用；正確性要求觀察到的結果與原程式契約相符。

若張量共享 storage，`x.add_()` 的影響可能透過另一個 view 被觀察到。
因此不能單純把所有 inplace 換成 out-of-place 就宣布等價。

**Repo：** [remove_copy_pass.py](tensorrt_llm/_torch/compilation/remove_copy_pass.py) 與
[Backend.optimize](tensorrt_llm/_torch/compilation/backend.py) 的 `remove_copy_for_mutates_args`。
這是 checkout 中與 mutation/copy 優化相關的入口；不是宣稱它完整實作 PyTorch functionalization。

**練習：** 畫一個 tensor 與其 view，說明原地更新如何影響另一個參照。

## 09｜第 06 集主題：Inductor IR 與中間值

**問題：** 圖裡的每個節點，是否都需要一份獨立的 GPU buffer？

IR 是編譯器保存計算意義的中間表示。對逐元素組合，編譯器可以表達「輸出索引 i 的值怎樣由輸入算出」。
如果相鄰運算適合融合，中間值可留在 kernel 內，不必都 materialize 成顯存張量。
但 reduction、外部 library 呼叫、跨節點重用與 layout 要求，可能使 intermediate 仍需要實體 storage。

FX graph、ATen graph、Inductor IR 與最後生成的 kernel code 處於不同層次，不能當成同一張圖。
看 graph 節點數量也不能直接推算 kernel launch 次數。

**Repo：** [backend.py](tensorrt_llm/_torch/compilation/backend.py) 第 183 行的 `compile_fx_inner`；
[piecewise_optimizer.py](tensorrt_llm/_torch/compilation/piecewise_optimizer.py) 的 `compile_fx`。
它們是進入 Inductor 的接點；Inductor 本體在 PyTorch dependency，不在 TRT-LLM 這兩個檔案裡。

**練習：** 對同一段小函式分別列出 Python operations、概念圖節點與預期 kernel 邊界，最後查看 generated code 核對。

## 10｜第 07 集主題：Compiler Scheduling 與融合限制

**問題：** 兩個相鄰算子，什麼時候能放進同一個 kernel？

編譯器必須考慮 read/write dependencies、shape、layout、同步與成本。
producer-consumer fusion 可能省 memory traffic；獨立運算也可能共享一次 launch。
但多 consumer、reduction、register 使用量、外部算子邊界與不同 parallelization 可能阻止或降低融合收益。

正確性先於效能：例如一個寫入會影響另一個讀取，改變順序就可能改變結果。
「能融合」與「融合後更快」是不同判斷。

**Repo：** [patterns/residual_add_norm.py](tensorrt_llm/_torch/compilation/patterns/residual_add_norm.py)
及 `Backend.build_custom_passes`：自訂 pattern 把特定模式換成融合操作。
這與 Inductor 自己的 scheduler 是相鄰層次，不能認為這個 pass manager 就是完整 Inductor scheduler。

**練習：** 找出 pattern 的匹配條件；推理為什麼 quantized 與非 quantized 路徑不能只改函式名稱。

## 11｜第 08 集主題：Triton 的 Program 與 Block

**問題：** 怎麼用 Python 形式描述 GPU 上一批資料的運算？

一個 Triton program 處理一個資料區塊，`program_id` 決定區塊位置，`arange` 建立區塊內的索引，mask 保護尾端。
Triton 的邏輯元素區塊不是「一個元素對一個 CUDA thread」的固定映射；實際分配由編譯器與配置決定。
`BLOCK=4096` 不代表一個 CUDA block 啟動 4096 threads。
[Triton 官方入門](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)。

**Repo：** `_relu2_kernel` 使用 `tl.load` → fp32 計算 → cast → `tl.store`，有明確 NaN propagation。
外層 `fused_relu2` 還檢查 output layout 與 partial overlap，這些屬於 operator correctness，不能只讀五行 kernel 就略過。

**練習：** 設 n=10、BLOCK=4，列出三個 programs 的索引與 mask。這是教學參數，不是 repo 的固定 launch 配置。

## 12｜第 09 集主題：編譯快取不是一種快取

**問題：** 為什麼重啟 Python 後，有時仍要 warmup 或等待？

| 快取／狀態 | 保存什麼 | 能否直接取代其他層 |
|---|---|---|
| Dynamo runtime variants | compiled code 與 guards 等程序內狀態 | 不能等同已 capture 的 CUDA Graph |
| 圖／codegen artifact cache | 適用於特定條件的編譯產物 | 不保證所有 shape 與版本命中 |
| Triton / library JIT cache | 特定 specialization 的 kernel 產物 | 不代表 serving resource 已配置 |
| Autotune cache | 工作負載下選出的 tactic／配置 | 不代表所有 GPU 上都最佳 |
| CUDA Graph execution state | captured graph、相關 storage 與執行狀態 | 不等同一般磁碟 code cache |
| KV / prefix cache | request 的模型中間狀態 | 與編譯快取用途不同 |

是否跨程序保存、如何失效與 key 的內容，必須逐層看版本和實作。
重啟後 artifact cache 命中，仍可能需要載入、配置、metadata preparation 或 capture。
[PyTorch 編譯快取教學](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)。

**Repo：** [flashinfer sampling 的 prewarm](tensorrt_llm/_torch/pyexecutor/sampler/ops/flashinfer.py) 第 381 行附近，將 lazy JIT 成本與正常呼叫區分。

**練習：** 將啟動、第一次請求、穩態請求分開計時；重啟程序再測，不以同一程序第二次呼叫代替 cold-start 實驗。

## 13｜第 10 集主題：GEMM Templates 與 Autotune

**問題：** 矩陣乘法明明公式相同，為什麼有不同 kernel？

`C[M,N] = A[M,K] × B[K,N]` 的 M/N/K、dtype、layout、scales 與 GPU 會影響最佳 tile、pipeline 與實作。
Autotune 在候選中量測或使用策略挑選，並付出搜尋、編譯、workspace 與記錄成本。
它不是數學上保證全域最佳，也不一定每次 forward 重跑搜尋。

Tensor Core 可加速支援格式的矩陣運算，但小 shape、低重用、layout 轉換或其他 overhead 仍可能主導。
GEMM 的 arithmetic intensity 要把實際 bytes 與 reuse 算進去，不只看 FLOPs。

**Repo：** [linear.py](tensorrt_llm/_torch/modules/linear.py) 第 585、662 行附近的 cuBLAS 分支，
以及 [autotuner.py](tensorrt_llm/_torch/autotuner.py)。這兩個入口不代表所有 Linear 都共用同一套 autotune。

**練習：** 為小 batch decode 與大 prefill 列出不同的 M，推理同一 tactic 為什麼可能不適用。

## 14｜第 11 集主題：CUDA Graphs 省掉什麼

**問題：** 每輪 decode 重複提交很多小 GPU 工作，CPU 提交成本能否降低？

Graph capture 記錄可重播的 GPU 工作與 dependencies；replay 降低 host 端逐項提交成本。
GPU 仍要執行那些運算，不會因 replay 就免除 memory traffic 或把所有 kernel 變成一個。
kernel fusion、Graph replay 與 overlap 分別處理不同成本，可以共存。

普通 capture/replay 需要滿足對執行結構、storage 位址與操作相容性的要求。
更新同一 storage 的資料值，與每次重新配置 tensor 並換掉 pointer，是不同操作。
若修改 graph 參數或使用其他進階更新機制，必須有相應 API 與限制，不能假設一般 replay 會自動接受。
[PyTorch CUDA Graph 語意](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs)。

**Repo：** [CUDAGraphRunner.capture / replay](tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py) 第 686、789 行。
[PiecewiseRunner.__call__](tensorrt_llm/_torch/compilation/piecewise_optimizer.py) 第 199 行會記錄 input addresses、warmup、capture，再 replay。

**練習：** 畫出「持久 input buffer → 寫入新資料 → replay」；標出哪個 owner 必須保證 storage 仍存活。這就接回 M01。

## 15｜第 12 集主題：Triton、CUTLASS、cuBLAS、CuTe DSL

**問題：** 這些名字不是同一層的替代選項，該如何比較？

| 工具 | 主要角色 | 通常需要使用者決定什麼 |
|---|---|---|
| CUDA C++ | 通用 GPU programming | threads、memory、同步、launch 與演算法 |
| Triton | block/tensor-oriented GPU language 與 compiler | 區塊運算、layout/tiling、launch/autotune 配置 |
| cuBLAS / cuBLASLt | 可直接呼叫的線性代數 library | dimensions、strides、types、workspace、algorithm 選項 |
| CUTLASS | 可組合的 CUDA templates 與 kernel building blocks | tile、MMA、pipeline、epilogue 與硬體策略 |
| CuTe / CuTe DSL | layout/tensor 抽象與 Python GPU kernel 開發工具 | 資料到 threads 的配置、copy/MMA、同步與 pipeline |
| FlashAttention | attention 演算法家族及實作 | attention 語意與支援的 kernel 介面 |
| FlashInfer | 面向 inference 的 operator library | attention metadata、layout、workspace 與實際 backend |

Triton 和 CuTe DSL 都能由 Python 形式撰寫 GPU 程式，但抽象、layout 表達與硬體控制方式不同。
cuBLASLt 可以選演算法與 epilogue，CUTLASS/CuTe 可以用來組合更特定的實作；實際選擇看需求與量測。
這些元件可在同一次 model forward 中同時出現。
[cuBLAS 官方文件](https://docs.nvidia.com/cuda/cublas/index.html)、[CuTe DSL 官方概覽](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)。

**Repo：** `requirements.txt` 包含 FlashInfer、Triton 與 `nvidia-cutlass-dsl`；
[cute_dsl_kernels](tensorrt_llm/_torch/cute_dsl_kernels/) 與 [CUDA kernels](cpp/tensorrt_llm/kernels/) 是不同實作入口。

**練習：** 解釋「PyTorch 模型 → custom op → FlashInfer → 某個 GPU kernel」的每一層責任；不要假設 FlashInfer 所有 kernel 都用 Triton 寫。

## 16｜第 13 集主題：動態 Request 如何重用 Graph

**問題：** request 數量與 KV 長度一直變，怎樣重用較穩定的 GPU execution pattern？

常見策略包括 capture 多個尺寸、以 bucket/padding 選擇、更新持久 metadata，或把不適合 capture 的部分留在 eager。
例如教學 buckets `[1,2,4,8]` 中三筆 decode requests 可選四筆容量，第四筆是受控制的 padding。
必須保證 dummy 資料、mask、KV 資源與輸出選取正確；不能把多算一筆當作無條件安全。

batch size 相同仍可能有不同 KV 長度或 metadata；compiler 的符號 shape 支援也不自動保證同一份 CUDA Graph 可用。
此段是通用策略與 TRT-LLM 接點，不是已驗證講者的 vLLM 範例或當前 vLLM 全部實作。

**Repo：** [engine/cuda_graph.py](tensorrt_llm/_torch/pyexecutor/engine/cuda_graph.py) 與
[piecewise 文件](docs/source/features/torch_compile_and_piecewise_cuda_graph.md)。
本 checkout 區分 generation-only graph 與 context/mixed 的 prefill graph，不能只看一份 capture list。

**練習：** 固定兩筆 requests，讓其中一筆 KV 從 31 變成 33 tokens；推理 page count、metadata、plan 與 Graph 哪些可能需要更新。

## 17｜FlashInfer：它在 TRT-LLM 裡究竟做什麼

**一句話：** TRT-LLM 組織整個推論流程，FlashInfer 可提供其中某些高效 GPU operators。

不是只有 attention。在本 checkout 可以直接看到：

| 接點 | 真實入口 | 往上服務誰 |
|---|---|---|
| Attention metadata 與執行 | [attention/backends/flashinfer.py](tensorrt_llm/_torch/attention/backends/flashinfer.py) | attention 模組、model forward |
| Activation、norm、RoPE wrapper | [custom_ops/flashinfer_custom_ops.py](tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py) | MLP、normalization、位置編碼與 compile |
| Sampling | [sampler/ops/flashinfer.py](tensorrt_llm/_torch/pyexecutor/sampler/ops/flashinfer.py) | logits/probs 處理與生成 token |
| MoE kernel 適配 | [fused_moe/trtllm_gen](tensorrt_llm/_torch/moe/fused_moe/trtllm_gen/) | 專家運算；具體支援看各檔與 backend |
| 可用性檢查 | [flashinfer_utils.py](tensorrt_llm/_torch/flashinfer_utils.py) | import、環境與功能接點 |

`attn_backend` 選的是 attention backend 家族，不是整個模型所有算子的 library。
本 checkout 的 [Attention Guide](tensorrt_llm/_torch/attention/ATTENTION_DEVELOPER_GUIDE.md) 列出 `TRTLLM`、`VANILLA`、`FLASHINFER` base families。
`TRTLLM` attention 路徑也有 [FlashInfer trtllm-gen 適配](tensorrt_llm/_torch/attention/backends/fmha/flashinfer_trtllm_gen.py)。
所以看到 `TRTLLM` 不足以推斷「完全沒有 FlashInfer」；要往下追實際 dispatch。

同樣地，FlashInfer wrapper 內部的 `backend` 參數，與 TRT-LLM 的 `attn_backend` 不在同一層。
目前檔案可見 `fa2`、`trtllm-gen` 等條件，但支援組合須看 GPU、dtype、head dim 與版本，不能任意互換。
[FlashInfer Attention API](https://docs.flashinfer.ai/api/attention.html)。

## 18｜FlashInfer 的 Plan / Run 與 Paged KV

**架構位置：** request scheduler 已選好工作與 KV 資源，attention adapter 將它們轉成 kernel 能理解的 metadata。

`plan` 準備對應 workload 的執行安排與 metadata；`run` 使用 Q/KV 與相關資料計算 attention。
Plan 不是分配整個 serving request 的 scheduler，也不是 Dynamo tracing，更不是 CUDA Graph capture。
何時能重用 plan、何時需要刷新，取決於 backend、metadata、KV 長度和 Graph 模式。

**可直接追的本地程式：**

1. `FlashInferAttentionMetadata`：第 309 行。
2. `prepare`：第 1620 行；檢查每 iteration 準備哪些資訊。
3. prefill `plan`：第 2224 行；decode `plan`：第 2287 行。
4. `FlashInferAttention`：第 2329 行。
5. `forward_impl`：第 2617 行，再沿分支讀 `.run(...)`。

這些行號來自本次 checkout 的 [flashinfer.py](tensorrt_llm/_torch/attention/backends/flashinfer.py)；不是所有 request 都走同一段。

### 用兩個 Requests 手算一次

假設每頁 16 tokens，A 的有效 KV 長度為 38，B 為 9。以下是示意 page-list 格式，不是所有 TRT-LLM backend 的實體配置。

```text
request             A                 B
logical pages       0, 1, 2           0
physical page IDs   7, 2, 9           5
indptr              [0, 3, 4]
indices             [7, 2, 9, 5]
last_page_len       [6, 9]
length              2×16+6 = 38       0×16+9 = 9
```

`indptr` 把扁平 page list 分給每筆 request；`indices` 指向實體頁；尾頁長度防止讀到無效位置。
若每筆 query 長度為一，普通 decode 的 Q 可概念化為 `[2,Hq,D]`；prefill 則可能有 ragged query offsets。
`Hq` 不一定等於 `Hkv`，GQA 的 mapping 必須正確。

FlashInfer 文件的典型 combined paged KV 在 NHD 格式可寫成 `[pages,2,page_size,Hkv,D]`，
但 tuple K/V、HND、量化與本地 allocator layout 都可能不同；先檢查 adapter 的 stride/layout 契約。
[KV layout 官方說明](https://docs.flashinfer.ai/tutorials/kv_layout.html)。

**檢查：** B 完成後把實體頁 5 分配給另一筆 request，哪些舊 metadata 與未完成 GPU work 必須先處理？

## 19｜Custom Op 與 Fake Implementation：如何接上編譯器

**問題：** 外部高效 kernel 不能被 Python tracing 完整看穿，怎麼放進模型圖？

Custom op 提供可辨識的 operator 邊界。Fake implementation 提供編譯期所需的輸出 metadata，
不是偷偷在 CPU 上算一遍 attention，也不是「假結果可以拿來 sampling」。
mutation schema、aliasing 與 output shape/dtype 必須和真實行為一致。

**本地例子：** [flashinfer_custom_ops.py](tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py) 第 16 行：

```python
@torch.library.custom_op("trtllm::flashinfer_silu_and_mul", mutates_args=())
def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    return silu_and_mul(x, enable_pdl=get_env_enable_pdl())
```

其 `register_fake` 回傳對應一半最後維度的 metadata。這反映 gated activation 的輸出契約，不能直接套到任意算子。
`flashinfer_fused_add_rmsnorm` 則宣告修改 `input` 與 `residual`，與無 mutation 的 activation 不同。

Opaque custom op 可讓 compiler 保留這個節點，但一般不能任意跨越邊界分析或融合其內部。
因此可搭配 pattern pass，或由 library 直接提供 fused operator。
[PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)。

**檢查：** 如果 fake implementation 的輸出 shape 正確，但 mutation 宣告錯誤，為什麼程式仍可能算錯？

## 20｜Piecewise CUDA Graph：編譯與 Library 的接縫

**問題：** 模型大部分可重播，但某些 attention planning 或 metadata 行為需要每輪處理，怎麼辦？

Piecewise 思路是保留需要 runtime 處理的邊界，將相容區段 capture/replay。
在本 checkout 的實作中，`_piecewise_boundary_ops` 列出 attention、MLA、GDN、Mamba 等可選 custom op 邊界。
不能推論「所有 attention 永遠不可 capture」，因為普通 generation graphs 與其他 backends 可能支援不同方式。

本地概念路徑：

```text
torch.compile → TRT-LLM Backend → AOT 圖整理與 patterns
              → 依 boundary ops 切段
              → supported segment: compile / warmup / capture / replay
              → excluded boundary: 對應 eager/custom op callable
```

[PiecewiseInterpreter / PiecewiseRunner](tensorrt_llm/_torch/compilation/piecewise_optimizer.py) 會維護 entries、capture 狀態與 storage 位址。
`PiecewiseRunner.__call__` 不符合 entry 或 flags 時會走 `default_callable`，不是無條件 replay。

本 checkout 文件也有不透過 torch.compile 的實驗性 `breakable` prefill graph 路徑。
它與 piecewise 是不同選項，支援限制要查文件，不能把「不用 compile」說成「不用 Graph」。
[本地 compile / prefill graph 文件](docs/source/features/torch_compile_and_piecewise_cuda_graph.md)。

**檢查：** Symbolic shape 已能處理多種 token 數，為何還會維護多份 captured graph entries？

## 21｜把資源生命週期接回 M01

上面每個效能技巧都依賴正確的資源管理：

| 資源 | 至少要問的 lifetime 問題 | 讀 code 的方向 |
|---|---|---|
| Input / output tensors | replay 時 storage 是否仍存在、位址是否符合要求 | graph runner、持久 buffers |
| Workspace | 是否被另一個 concurrent op 提早重用 | FlashInfer wrappers、metadata、stream dependencies |
| KV pages | request 結束時是否仍有 GPU work 讀寫 | KV manager、request completion、events |
| CUDA stream | 使用它的 manager／allocator 是否仍需要它 | BufferManager、CudaStream、allocator ownership |
| Compiled callable / graph | 捕獲資源由誰保有，reset 時如何處理 | Backend、PiecewiseRunner、graph pools |

`unique_ptr` / `shared_ptr` 描述 C++ object ownership；stream/event 描述 GPU work 的 ordering 與 completion。
兩者都重要：一個 C++ object 被保住，不等於跨 streams 自動同步；host object 被析構，也不等於之前 GPU 工作都已完成。

**M01 接續題仍只有一題：** 兩個 `BufferManager` 共用 stream、各有 buffers，為什麼 stream 可能用 `shared_ptr`，回傳 buffer 可以用 `unique_ptr`？
這份教材沒有替你回答或勾選進度。

## 22｜怎樣驗證整套優化有效

### 先確認 Correctness

在相同模型、weights、tokenization 與 sampling 設定下比較；operator 級則固定 shape、dtype、layout 與語意。
處理尾端、空輸入支援範圍、極端值、NaN、aliasing、mutation 與 page boundary。
Sampling 含隨機性，不能只用一次 token 完全相同作為所有分布正確性的證據。

### 再把時間分層

```text
啟動：load + allocation + JIT/compile + autotune + capture + warmup
TTFT：queue + 輸入處理 + admission + prefill + first sampling/response
後續 token：scheduling + metadata + forward + sampling + 必要等待與交付
```

這是成本分類，非在 overlap 系統裡可直接相加的時間等式；量測邊界與重疊區域要在 profiler 裡確認。
比較 host wall time、CUDA events 與 serving latency，先說清楚各量到哪一段。

| 症狀 | 優先查看 | 可能動手的層 |
|---|---|---|
| Python 重啟後第一筆很慢 | load、JIT、cache miss、capture、warmup | initialization / compile cache |
| shape 變化時延遲尖峰 | guards、recompiles、graph entry/fallback | Dynamo / graph selection |
| GPU 兩次工作間有空白 | CPU scheduler、metadata、launch、同步 | runtime / planning / Graph |
| 大量 intermediate traffic | generated code、fusion boundaries | compiler / fused custom ops |
| attention 隨 KV 變長而變慢 | page layout、bytes、plan、kernel、batch | KV / FlashInfer / attention |
| kernel 快了端到端沒變 | 佔比、queueing、其他 stage 與額外成本 | 全流程 |

可用 `TORCH_LOGS=graph_breaks,recompiles,guards` 先查編譯問題，需看 generated code 時再加 `output_code`。
這些開關可能產生大量 logs，應用於小 lab；欄位以安裝版本支援為準。
[PyTorch observability](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.observability.html)、
[本 repo Performance Analysis](docs/source/developer-guide/perf-analysis.md)。

本文件的圖、traffic 推導與 buckets 都是教學材料。**尚未在本機執行模型或 GPU 效能實驗，沒有 speedup 結論。**

## 23｜醒來後如何開始：三次短課

**第一課：把 Request 放上架構圖。** 讀 01～03，用自己的話指出 serving、compiler、kernel 三層。一次只回答一個定位問題。

**第二課：接續 M01。** 打開 `bufferManager.h` 的 aliases 與 constructor，回答目前待答的 ownership 題；再追一次具體資源建立與保存。

**第三課：做一個小的融合例子。** 讀 04 與 11，自己列出 ReLU² 的 traffic 與索引。之後在獨立 lab 驗證，再比較 eager/compiled/手寫 kernel。

後續依序接 guards → custom op/fake → FlashInfer plan/run → CUDA Graph/piecewise，
每次在上層 request 流程中標出目前正在理解哪個接縫。
唯一完成清單仍在課程地圖，閱讀此處不自動代表 M00/M01/M08/M09 完成。

## 24｜帶著問題讀 Repo

以下是各節複習題的索引，互動帶讀時一次只選一題，不同時全部要求回答。

| 複習題 | 應回到的節次 |
|---|---|
| torch.compile 與 CUDA Graph 分別省哪一段成本？ | 03、14、20 |
| FlashInfer 在哪裡接到 request 的 KV pages？ | 17、18 |
| 假如 attention 用 FlashInfer，MLP 還能用 cuBLAS 或 Triton 嗎？ | 01、15、17 |
| fake op 知道 shape，為什麼不等於它算出了 logits？ | 05、19 |
| 同一 batch size，KV 長度改變時還要檢查什麼？ | 16、18、20 |
| compiled code cache 命中，為什麼仍可能重新 capture？ | 12、14 |
| 為何一個 Graph 裡可以包含多個 library kernels？ | 14、15 |
| 怎樣證明你找對了實際 dispatch，而非只讀到候選 class？ | 02、17、18 |
| 如何從 buffer ownership 一路解釋 Graph replay correctness？ | 14、20、21 |

## 25｜本地原始碼導航

行號為 2026-10-27 當次 checkout 的查閱結果；更新後以 symbol 搜尋為準。

| 檔案 | Symbol / 查閱位置 | 要核對的事 |
|---|---|---|
| [model_engine.py](tensorrt_llm/_torch/pyexecutor/model_engine.py) | compile 設定 434；`Backend` 建構 490；`torch.compile` 511 附近 | 選定模型的 compile 路徑與分支 |
| [backend.py](tensorrt_llm/_torch/compilation/backend.py) | `Backend` 43；`__call__` 187；`aot_module_simplified` 222 | 自訂 backend 到 AOT/Inductor |
| [piecewise_optimizer.py](tensorrt_llm/_torch/compilation/piecewise_optimizer.py) | boundaries 26；`PiecewiseRunner` 148；`__call__` 199 | 切段、warmup、capture、fallback/replay |
| [flashinfer.py](tensorrt_llm/_torch/attention/backends/flashinfer.py) | metadata 309；prepare 1620；plan 2224/2287；backend class 2329 | page metadata 到 attention run |
| [flashinfer_custom_ops.py](tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py) | activation custom op 16 與 fake registration | 真實算子與 tracing metadata |
| [sampler FlashInfer ops](tensorrt_llm/_torch/pyexecutor/sampler/ops/flashinfer.py) | top-k/top-p 105 附近；prewarm 381 附近 | attention 之外的 FlashInfer 使用 |
| [cuda_graph_runner.py](tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py) | `CUDAGraphRunner` 156；capture 686；replay 789 | 普通 generation graph 管理 |
| [fused_relu2_triton.py](tensorrt_llm/_torch/fused_relu2_triton.py) | kernel 26；wrapper 46 | 手寫融合、layout、aliasing |
| [linear.py](tensorrt_llm/_torch/modules/linear.py) | cuBLAS 路徑 585、662 附近 | 模型 Linear 與 library dispatch |
| [bufferManager.h](cpp/include/tensorrt_llm/runtime/bufferManager.h) | aliases 43；constructor 54 | ownership 與 GPU 資源管理 |

再往外部 library 追時，記下 package version、實際函式與 backend；不能只用最新 main 的實作推定這份 checkout 的行為。

## 26｜FlashAttention、PagedAttention 與 FlashInfer 的關係

**先回到公式：** 對一個 query head，attention 是 `softmax(QKᵀ / sqrt(D) + mask) V`。
Q 的 query 長度和 K/V 的歷史長度可不同；GQA 還涉及 query heads 到 KV heads 的對應。

### FlashAttention：改變資料搬運與計算安排

直接實作可能保存整份 query×key score/probability 矩陣。
FlashAttention 以 tiles 計算，配合 online softmax 維護局部統計與輸出，減少顯存往返與大型中間矩陣。
它仍依 exact attention 語意處理資料；dense prefill 的主要計算複雜度不會因此從二次自動變線性。
浮點運算順序不同，結果不保證 bitwise 一致。
[FlashAttention 原始論文](https://arxiv.org/abs/2205.14135)。

單一 query row 的教學遞迴如下。新 tile 的 scores 已包含 scaling/mask，舊狀態為 max `m`、分母 `l`、未正規化向量 `o`：

```text
m_new = max(m, max(scores_tile))
alpha = exp(m - m_new)
p     = exp(scores_tile - m_new)
l_new = alpha * l + sum(p)
o_new = alpha * o + p @ V_tile
最後 output = o / l
```

這不是完整 kernel：還要處理初始值、完全 masked rows、GQA、並行歸約、dtype 與記憶體配置。
重點是：改變目前 max 時，用同一比例重縮舊累積值，因此不需要保留所有舊 scores。

### PagedAttention：改變 KV 的配置與尋址

KV 以 blocks/pages 管理，透過邏輯到實體映射存取；可降低配置碎片並支援相應的 reuse 策略。
它不會讓有效 KV tokens 的元素數憑空變少，也不等於 attention softmax 被近似。
對應 [本 repo KV cache 文件](docs/source/features/kvcache.md) 與第 18 節的 page list。

### FlashInfer：提供與整合這些需求相關的 Operators

Library 必須把不同 KV layout、變動長度、query 形狀與 GPU 特性轉成可執行的計畫。
這就是第 18 節的 metadata、plan/run 與不同 backend 存在的原因。
FlashInfer 論文討論 KV 格式、JIT 客製化與動態負載下的排程及 Graph 相容性；
paper 數據不當作目前本機或 TRT-LLM 版本的 benchmark。
[FlashInfer 論文](https://arxiv.org/abs/2501.01005)。

**檢查：** 可以同時使用 paged KV 與 IO-aware tiled attention 嗎？它們各改變什麼？

## 27｜把兩個 Requests 從 Prefill 追到第一輪 Decode

這是手算模型，不是某個已執行的 checkpoint：32 layers、Hq=32、Hkv=8、D=128、KV 為 BF16，
單 GPU、無 prefix sharing、無特殊壓縮或輔助 cache。A prompt 38 tokens，B prompt 9 tokens，每頁 16 tokens。

### Prefill：一次處理已知的 Prompt

以 packed tokens 表示時，本輪共 47 個 query tokens。隱藏維度示意為 4096，
Q 可視為 `[47,32,128]`，新 K/V 各為 `[47,8,128]`；具體 forward 輸入與 layout 要核對模型。
A 和 B 各自套用正確的 causal 邊界，不能把 packed storage 當作一條 47-token 序列互相注意。

每 layer 的有效 KV bytes：

```text
2 × 47 × 8 × 128 × 2 = 192,512 bytes = 188 KiB
```

頁數為 A 三頁、B 一頁；若以完整四頁保留空間，則每 layer 是：

```text
2 × (4 × 16) × 8 × 128 × 2 = 262,144 bytes = 256 KiB
32 layers 合計 8 MiB
```

未包含 metadata、workspace、allocator 預留與其他模型資源；與「有效 KV 元素數」分開計算。

### First Sampling：預測第一個新 Token

普通生成可從各 request 最後有效 prompt 位置的 logits 產生第一個新 token。
這時 KV cache 包含 prompt；**剛 sample 出的新 token 還沒有因為 sampling 自動生成 K/V**。
產生 token 的數學操作，與對該 token 執行下一次 model forward，是不同步驟。

### Decode：把剛生成的 Token 當下一輪輸入

假設兩筆都沒結束，下一輪各輸入一個 token，Q 共兩列。
本次 forward 為這兩個新輸入產生 K/V；各 request 的有效 KV 長度由 38/9 增至 39/10，
attention 讀取各自歷史並產生下一個位置的 logits，再做 sampling。
具體 cache write 與 attention kernel 的融合方式、執行順序由 backend 決定。

普通單 token decode 的 attention 會讀歷史 KV；整個模型仍包含 projection、MLP 等運算。
小 batch 時 weights/KV traffic 或 launch overhead 常重要，但大 batch、長 context、量化與不同硬體可改變瓶頸。
不能把「decode 常 memory-bound」當作所有 kernel 的固定結論。

### 完成、取消與 Reuse

若 B sample 到 EOS 或符合停止條件，它不再需要為該完成狀態執行下一輪普通 decode。
runtime 要處理 pending GPU work、回應狀態與 KV 釋放／reuse 政策。
釋放邏輯 ownership 不能造成仍在使用的實體頁被過早分配給新 request。

**檢查：** 第一個新 token 已經回應給使用者，為什麼它的 K/V 可能尚未在 cache 裡？

## 28｜擴展到多 GPU：哪幾層開始改變

先保持前面的 request 主線，再加入以下變化；不要一次把所有平行方式當作同一種切模型。

| 方式 | 主要切分對象 | 往端到端流程增加的問題 |
|---|---|---|
| Tensor Parallelism（TP） | 一層內的 tensor／計算 | head/weight 分片、collectives、KV 分布、通訊與算子的融合 |
| Pipeline Parallelism（PP） | 模型 layers / stages | activation 傳遞、microbatches、stage scheduling 與空泡 |
| Expert Parallelism（EP） | MoE experts | token routing、dispatch/combine、負載與通訊 |
| Prefill/Decode 分離 | request 的不同執行階段 | KV transfer、交接狀態、排隊與不同 GPU 資源配置 |

NCCL 提供 GPU 通訊 primitives；NIXL 在本 repo 的 disaggregated serving 可參與 KV transfer。
這些元件不代替 tokenizer、model 或 FlashInfer attention operator；它們提供不同的系統能力。
通訊本身可能涉及 kernel、stream、workspace 與 capture 相容性，因此也要接回 M01/M06。

閱讀順序：[parallel strategy](docs/source/features/parallel-strategy.md) →
[MoE guide](tensorrt_llm/_torch/moe/fused_moe/MOE_DEVELOPER_GUIDE.md) →
[disaggregated serving](docs/source/features/disagg-serving.md)。
在選定配置下追一次實際 communicator 與 cache transceiver，再量測 compute / communication / overlap。

**檢查：** attention kernel 快 20%，如果 request 主要卡在 KV transfer，端到端延遲會自動快 20% 嗎？
