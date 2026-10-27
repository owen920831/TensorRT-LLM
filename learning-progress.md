<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM 學習紀錄

課程、教學規則與唯一完成清單：[interview-preparation.md](interview-preparation.md)。
本文件只維護學習狀態、實際回答與證據，不複製另一份課程清單。

## 目前狀態

- 最後更新：2026-10-27。
- Checkout：`e921fa39b07c669f2633ac95390fd9deac32d92b`。
- 目前課程：M01，C++ RAII 與 GPU buffer ownership。
- 整體位置：executor/model 執行所需的 GPU 資源管理；目前是選讀案例，尚未追通特定模型的完整 allocation 呼叫鏈。
- 閱讀範圍：`bufferManager.h` 第 43～54 行的 aliases 與 constructor 宣告；
  `iBuffer.h` 第 369～375 行的 pointer aliases；`bufferManager.cpp` 第 33～38 行的 constructor。
- 已提出的問題：兩個 `BufferManager` 共用一條 CUDA stream、各自配置不同 buffers 時，
  為什麼 stream 可能採用 `shared_ptr`，回傳 buffer 可以採用 `unique_ptr`？
- 學習者回答：尚未收到；之後的訊息是在調整學習方式，不能當成已回答。
- 待評估觀念：單一／共享 ownership、持有者與釋放時機的關係；尚未據此判定理解程度。
- 練習：尚未開始。GPU correctness、benchmark 與 profiler 均未執行。
- 下一步：以繁體中文、一題一答的方式接續這題；回答後再決定是否補 ownership，或追 constructor 的 `std::move`。
- 本課尚未驗收，不勾選 M01。

## 課程紀錄格式

教材更新：已加入 [端到端概念筆記](compiler-and-flashinfer-e2e.md) 與
[離線視覺化閱讀頁](e2e-learning.html)。下次先用架構圖確認 M01 的整體位置，再接續待答題。
影片僅取得主題 metadata，尚未取得逐字稿；來源限制見 [紀錄](bilibili-compiler-sources.md)。
教材產出不代表學習者已學會，保留上面的待回答與未驗收狀態。

每完成一個小節追加一筆，保留學習者原始回答，不改寫成其未曾表達的正確答案。
若只是提示後理解，需另有無提示解釋或新情境驗證，才判定掌握。

```text
日期 / 任務 / Checkout：
架構位置、上游與下游：
閱讀的 path / symbol / 當下行號：
本輪問題：
我的原始回答：
指出的缺口與給過的提示：
我的修正解釋：
練習與 correctness 證據（未跑就寫未驗證）：
面試追問 / 換情境驗證：
結果（待回答 / 需提示 / 可獨立解釋 / 已完成實作驗收）：
下一題：
```
