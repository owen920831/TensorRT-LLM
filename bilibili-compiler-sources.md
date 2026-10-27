<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Bilibili 合集來源紀錄：torch 編譯器
查閱日期：2026-10-27。作者：Cishoon（mid `106695488`）；season `9198587`。
[原始合集](https://space.bilibili.com/106695488/lists/9198587?type=season)。
當次清單共 13 集，合計 3 小時 29 分 16 秒。這是 metadata 快照，非觀看紀錄。
## 取得方式與限制
- 公開合集 API 回傳成功，可核對作者、合集簡介、影片標題、BV ID 與片長。
- [合集清單 API](https://api.bilibili.com/x/space/fav/season/list?season_id=9198587&pn=1&ps=100)。
- [交叉核對 API](https://api.bilibili.com/x/polymer/web-space/seasons_archives_list?mid=106695488&season_id=9198587&sort_reverse=false&page_num=1&page_size=100)。
- 合集網頁顯示驗證頁；影片網頁及 view API 回傳 HTTP 412。
- 第 08 集兩個 API 的標題有差異：一個為「Triton 與塊」的長標題，一個為「Triton 與 Codegen」。以下採收藏清單 API 的標題，BV ID 相同。
- 未取得可讀逐字稿，未聽看全部影片；不聲稱完整還原講者案例、時間點或結論。字幕取得狀態見文末。
- [端到端學習筆記](compiler-and-flashinfer-e2e.md) 是依本清單主題、官方文件與本地程式碼寫成的原創解說，與影片原話分開。
## 當次影片清單
| 集數 | 原始標題 | 片長 | 影片 |
|---|---|---|---|
| 01 | 【torch编译器】01.算子融合：GPU做加法时，99%的时间在摸鱼 | 15:08 | [BV149h16QE7p](https://www.bilibili.com/video/BV149h16QE7p/) |
| 02 | 【torch编译器】02.Dynamo抓图：把f跑了一遍，172亿次乘法一次没算 | 13:08 | [BV1n5hD69EW6](https://www.bilibili.com/video/BV1n5hD69EW6/) |
| 03 | 【torch编译器】03.guard守卫：换了10种形状，只编译了2次 | 13:43 | [BV158a46JEBs](https://www.bilibili.com/video/BV158a46JEBs/) |
| 04 | 【torch编译器】04.AOTAutograd：训练时，编译器提前把反向传播也编译好，并决定前向要给反向留下什么 | 13:49 | [BV1J8aY6CE7Q](https://www.bilibili.com/video/BV1J8aY6CE7Q/) |
| 05 | 【torch编译器】05.分解与函数化：PyTorch 的三千多个算子，编译器先拆成少数基础算子、去掉原地修改，不用一个个支持 | 13:38 | [BV1oSaE6YE6z](https://www.bilibili.com/video/BV1oSaE6YE6z/) |
| 06 | 【torch编译器】06.Inductor IR：编译器用一个函数写下输出的每个数怎么算，相邻的运算直接套进去，中间结果连显存都不用分配 | 14:32 | [BV13HaX6DE2K](https://www.bilibili.com/video/BV13HaX6DE2K/) |
| 07 | 【torch编译器】07.调度器与融合：编译器先弄清每个运算读写哪些数据，再把能一起算的放进同一个 kernel | 21:48 | [BV1P5aB6FENB](https://www.bilibili.com/video/BV1P5aB6FENB/) |
| 08 | 【torch编译器】08.Triton 与块：编译器把合并好的运算写成一段 GPU 程序，同时启动几千份，每份一次处理一整批数 | 15:51 | [BV1Nnay6JEbs](https://www.bilibili.com/video/BV1Nnay6JEbs/) |
| 09 | 【torch编译器】09.编译缓存：编译好的模型，重启Python还要再编译吗？ | 12:56 | [BV1C5Hh6SE1A](https://www.bilibili.com/video/BV1C5Hh6SE1A/) |
| 10 | 【torch编译器】10.矩阵乘模板与autotune：max-autotune 编译那么久，在做什么？ | 12:28 | [BV1gRHq62EK8](https://www.bilibili.com/video/BV1gRHq62EK8/) |
| 11 | 【torch编译器】11.CUDA Graphs：启动kernel的时间，能省掉吗？ | 10:40 | [BV115Hd6dEkb](https://www.bilibili.com/video/BV115Hd6dEkb/) |
| 12 | 【torch编译器】12. Triton、CUTLASS、cuBLAS、CuTe DSL...都是什么？ | 39:13 | [BV1w7p56qES3](https://www.bilibili.com/video/BV1w7p56qES3/) |
| 13 | 【torch编译器】13.vLLM：请求数一直在变，CUDA Graphs 还怎么用？ | 12:22 | [BV1WTJ26XEoB](https://www.bilibili.com/video/BV1WTJ26XEoB/) |

## 內容歸屬規則

1. 上表只能證明各影片標示的主題，不能證明講者用了哪些範例或逐步論證。
2. 「99% 時間」「只編譯 2 次」等標題文字，不當作普遍效能數據或本機測試結果。
3. FlashInfer 是學習者要求補充的內容；不能根據合集標題斷言第 12 集已講到它。
4. 若之後取得字幕或講者講義，再補「影片原文位置 → 筆記段落 → repo 接點」；不覆蓋原有來源區別。

## 匿名字幕查詢結果

另以每支影片的公開 pagelist API 取得 cid，再呼叫 player API。13 集 player API 均回傳 `code=0`，但 `subtitle.subtitles=[]`。

因此只知道本次匿名查詢沒有可取得的字幕清單；不能宣稱影片本身無字幕。行動版頁面也未提供可讀逐字內容。
