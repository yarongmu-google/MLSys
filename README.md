# Important Dates

| Date | Event |
| :---- | :---- |
| 01/22/2026 | Public launch • Call for participation • Open registration • Release starter kit |
| 02/09/2026 | Release 5 benchmark sets |
| 04/24/2026 | Track A and Track B submission deadline (11:59 PM PT) — **Closed** |
| 05/01/2026 | Writeup deadline (11:59 PM PT) — **Closed** |
| 05/11/2026 | Notify winners |
| 05/17/2026 \- 05/22/2026 | MLSys 2026 \- award ceremony |

# Results

The contest is closed and winners have been determined. Full top-10 leaderboards (per-benchmark and total scores) plus a Special Innovation Award technique-frequency breakdown are published here:

- [Track A leaderboard](./track_a_dashboard.md)
- [Track B leaderboard](./track_b_dashboard.md)

When a participant submitted multiple times, only the most recent submission counts. Winning teams have until 2026-07-11 to open-source their solution under MIT/Apache 2.0; code links will be added below as they become available.

## Track A: Systems Engineering — Winners

| Rank | Prize | Team | Score | Code |
|:---|:---|:---|---:|:---|
| First Place 🥇 | $10,000 | Vinci | 19.22 | [TBD](https://github.com/wenqi-deng/mlsys-2026-graph-scheduler) |
| Second Place 🥈 | $7,500 | VNRC | 18.52 | TBD |
| Third Place 🥉 | $5,000 | curling-grad | 18.32 | TBD |
| Special Innovation Award ⭐ | $2,500 | Jag | — | TBD |

## Track B: Agent Reasoning — Winners

| Rank | Prize | Team | Score | Code |
|:---|:---|:---|---:|:---|
| First Place 🥇 | $10,000 | TeamKorea | 20.97 | TBD |
| Second Place 🥈 | $7,500 | Scratchpad | 20.83 | TBD |
| Third Place 🥉 | $5,000 | Lazy Schedulers | 18.86 | TBD |
| Special Innovation Award ⭐ | $2,500 | iceburge | — | TBD |

Congratulations to all winning teams, and thanks to every participant for an exceptionally strong field of submissions.

# Problem Description

Please see [PROBLEM.md](https://github.com/yarongmu-google/MLSys/blob/main/PROBLEM.md).

# Submission Categories & Format

We are hosting two distinct tracks to celebrate both systems engineering and AI reasoning. Teams may participate in one or both tracks.

## Submission Mechanism

Submissions were accepted via the Official Submission Form. The submission window has closed.

* Deadline: April 24, 2026, 11:59 PM PT (closed).  
* File Naming: Please name your archive using the format: `TeamName_Track_SubmissionNumber.zip` (e.g., DeepRunners\_TrackA\_1.zip).  
* Form Settings Note: when uploading, ensure your file is a standard `.zip` archive.

## Track A: Systems Engineering (Human-Written)

This track focuses on the efficiency and robustness of the scheduling algorithm itself.

* Deliverable: A compiled binary executable named `mlsys`.  
* Interface: your binary must accept two command-line arguments:

```shell
$ ./mlsys <path_to_input.json> <path_to_output.json>
```

  * Note: The timeout is enforced by the organizer's harness, not passed as an argument. Your binary simply needs to write the solution file before it gets killed.  
* Zip Contents:  
  * `mlsys`: The statically linked binary (must run on Ubuntu 22.04 LTS).  
  * `source/`: Complete source code (C++/Rust/etc.) for verification.  
  * `writeup.pdf`: a 2-page summary of your approach.

## Track B: Agent Reasoning (AI-Generated)

This track benchmarks the ability of Gemini-powered Agents to solve scheduling puzzles. To ensure fair comparison, your agent must exclusively utilize the Google Gemini API for its reasoning and generation.

* Deliverable: a Python script named `agent.py`.  
* Interface: Your script will be executed as follows:

```shell
$ python3 agent.py <path_to_input.json> <path_to_output.json>
```

* Environment:  
  1. The organizers will inject the `GOOGLE_API_KEY` environment variable.  
  2. Network access is strictly blocked except to the Gemini API endpoint (`generativelanguage.googleapis.com`). Calls to any other LLM provider (OpenAI, Anthropic, etc.) will fail.  
  3. **Timeout:** Your agent script is strictly limited to **10 minutes** of execution time per benchmark (including all API calls and local processing).
* Zip Contents:  
  1. `agent.py`: The entry point script.  
  2. `requirements.txt`: A list of Python dependencies (e.g., google-genai, numpy).  
  3. `prompts/`: A folder containing your system prompts, few-shot examples, or agent definitions (required for reproducibility).  
  4. `writeup.pdf`: A short (1-2 page) summary of your prompting strategy and agent architecture.

Both tracks share the same set of 24 benchmark problems.

# Resources and Compute

To ensure fair play and broad accessibility, this contest is designed to be Zero Cost for all participants.

For Track A (Systems):

* Standard Hardware: since the problem is a specialized graph scheduling puzzle (JSON-to-JSON), no special hardware is required for the solution itself. A standard laptop or free Colab CPU instance is sufficient to run the `mlsys` binary and generate schedules.

For Track B (Agents):

* Free Intelligence: participants are encouraged to use Gemini 2.5 Flash via Google AI Studio. A free API key can be obtained at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). The free tier is sufficient for developing and testing agentic schedulers.

# Getting Started

To help you get started, we have released several C++ files in [this repository](https://github.com/yarongmu-google/MLSys) that define the basic problem and solution classes, along with various utilities. You are under no obligation to use these source codes.

| File | Description |
| :---- | :---- |
| [mlsys.h](https://github.com/yarongmu-google/MLSys/blob/main/mlsys.h) | Defines the Problem and Solution classes, along with methods for file I/O and solution evaluation. |

Please don't hesitate reaching out to the Contest Organizers if you encounter any trouble getting these materials to work.

# Benchmarks

The contest involved twenty-four industrial benchmarks; five were released in advance for participants to test against, and the remaining nineteen were withheld for evaluation. All twenty-four are now public in [`benchmarks/`](./benchmarks/).

**Note on Timeouts:** The timeouts listed below apply strictly to **Track A (Systems)**. For **Track B (Agents)**, a fixed timeout of **10 minutes** is applied to every benchmark to accommodate LLM inference latency.

| Benchmark Name | \# Nodes | \# Edges | Timeout | Released during contest? |
| :---- | :---- | :---- | :---- | :---- |
| [mlsys-2026-1](./benchmarks/mlsys-2026-1.json) | 5 | 9 | 2 second(s) | Yes |
| [mlsys-2026-2](./benchmarks/mlsys-2026-2.json) | 5 | 7 | 2 second(s) |  |
| [mlsys-2026-3](./benchmarks/mlsys-2026-3.json) | 4 | 6 | 2 second(s) |  |
| [mlsys-2026-4](./benchmarks/mlsys-2026-4.json) | 5 | 10 | 2 second(s) |  |
| [mlsys-2026-5](./benchmarks/mlsys-2026-5.json) | 19 | 34 | 5 second(s) | Yes |
| [mlsys-2026-6](./benchmarks/mlsys-2026-6.json) | 17 | 29 | 5 second(s) |  |
| [mlsys-2026-7](./benchmarks/mlsys-2026-7.json) | 15 | 21 | 5 second(s) |  |
| [mlsys-2026-8](./benchmarks/mlsys-2026-8.json) | 20 | 37 | 5 second(s) |  |
| [mlsys-2026-9](./benchmarks/mlsys-2026-9.json) | 32 | 56 | 15 second(s) | Yes |
| [mlsys-2026-10](./benchmarks/mlsys-2026-10.json) | 28 | 47 | 15 seconds(s) |  |
| [mlsys-2026-11](./benchmarks/mlsys-2026-11.json) | 26 | 38 | 15 seconds(s) |  |
| [mlsys-2026-12](./benchmarks/mlsys-2026-12.json) | 31 | 46 | 15 seconds(s) |  |
| [mlsys-2026-13](./benchmarks/mlsys-2026-13.json) | 63 | 126 | 30 seconds(s) | Yes |
| [mlsys-2026-14](./benchmarks/mlsys-2026-14.json) | 63 | 96 | 30 seconds(s) |  |
| [mlsys-2026-15](./benchmarks/mlsys-2026-15.json) | 61 | 97 | 30 seconds(s) |  |
| [mlsys-2026-16](./benchmarks/mlsys-2026-16.json) | 63 | 85 | 30 seconds(s) |  |
| [mlsys-2026-17](./benchmarks/mlsys-2026-17.json) | 103 | 198 | 60 seconds(s) | Yes |
| [mlsys-2026-18](./benchmarks/mlsys-2026-18.json) | 96 | 176 | 60 seconds(s) |  |
| [mlsys-2026-19](./benchmarks/mlsys-2026-19.json) | 103 | 154 | 60 seconds(s) |  |
| [mlsys-2026-20](./benchmarks/mlsys-2026-20.json) | 103 | 178 | 60 seconds(s) |  |
| [mlsys-2026-21](./benchmarks/mlsys-2026-21.json) | 152 | 280 | 120 seconds(s) |  |
| [mlsys-2026-22](./benchmarks/mlsys-2026-22.json) | 150 | 240 | 120 seconds(s) |  |
| [mlsys-2026-23](./benchmarks/mlsys-2026-23.json) | 121 | 186 | 120 seconds(s) |  |
| [mlsys-2026-24](./benchmarks/mlsys-2026-24.json) | 112 | 192 | 120 seconds(s) |  |

All twenty-four benchmarks (released-at-launch + previously-withheld) are now public in [`benchmarks/`](./benchmarks/). Each benchmark also has a `mermaid/` companion diagram for quick visualization.

# Evaluation and Scoring

The contest organizers will execute each team's binary across the nineteen withheld benchmarks on a dedicated 8-core Linux workstation with 32GB of RAM. Rankings will be established by calculating the total number of points per team, where points are derived by normalizing cost values against the best submitted solution.

For example, assume the following raw costs:

| Team | Benchmark X | Benchmark Y | Benchmark Z |
| :---- | :---- | :---- | :---- |
| Team A | cost: 100 | cost: 500 | cost: 800 |
| Team B | cost: 300 | cost: 200 | cost: 900 |
| Team C | cost: 600 | cost: 700 | cost: 400 |
| Min. Team Cost | cost: 100 | cost: 200 | cost: 400 |

The points that a team earns for a particular benchmark is then equal to min\_team\_cost / team\_cost

| Team | Benchmark X | Benchmark Y | Benchmark Z | Total Points | Ranking |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Team A | 1.000 points | 0.400 points | 0.500 points | 1.900 points | First Place 🥇 |
| Team B | 0.333 points | 1.000 points | 0.444 points | 1.778 points | Second Place 🥈 |
| Team C | 0.167 points | 0.286 points | 1.000 points | 1.452 points | Third Place 🥉 |

# Prizes and Awards

Thanks to our sponsor Google, we are offering a $50,000 Total Prize Pool, split evenly between the two tracks to encourage diversity in approaches.

| Track | Gold (1st) | Silver (2nd) | Bronze (3rd) | Special Innovation Award |
| :---- | :---- | :---- | :---- | :---- |
| Track A: Systems | $10,000 | $7,500 | $5,000 | $2,500 |
| Track B: Agents | $10,000 | $7,500 | $5,000 | $2,500 |

Award Rules:

1. Open Source: To be eligible for cash prizes, winning teams must open-source their solution (code or agent prompts) under an MIT/Apache 2.0 license within 2 months of the award.  
2. One Prize Per Team: A single team can compete in both tracks but may only claim the highest monetary prize earned across tracks (the unclaimed prize passes to the next rank).  
3. Agent Reproducibility: For Track B, the organizers will re-run your agent 3 times. The median score will be used to mitigate non-deterministic LLM outputs.

# Presentations

Teams who win the gold awards will be invited to present a \~10 minute informal overview of their approach at a special session held during the conference days. Presentations will be linked from the main MLSys competition website.

# Contest Eligibility

All are welcome to participate in the contest \-- including teams from academia, industry, and elsewhere \-- with the exception of the Contest Organizers and employees of the Contest Sponsor. Individuals are prohibited from participating in multiple teams. In order to be eligible for prizes, contestants must commit to releasing an open-source version of their implementation 1 month after winning (allowing teams sufficient time to write & submit papers detailing their approach).

# Frequently Asked Questions

To raise a question, please create an issue in this repository, or feel free to reach out to the contest organizers directly.

# Reference Materials

[ASPLOS 2025 contest track](https://github.com/asplos-contest/2025/blob/main/IOPDDL.md), which is a prequel of this problem.   
[Tensor processing primitives: a programming abstraction for efficiency and portability in deep learning workloads (2021)](https://dl.acm.org/doi/pdf/10.1145/3458817.3476206)  
[Apollo: Automatic Partition-based Operator Fusion through Layer by Layer Optimization (2022)](https://proceedings.mlsys.org/paper_files/paper/2022/file/e175e8a86d28d935be4f43719651f86d-Paper.pdf)  
[Optimus: An Operator Fusion Framework for Deep Neural Networks (2022)](https://dl.acm.org/doi/pdf/10.1145/3520142)  
[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022](https://openreview.net/pdf?id=H4DqfPSibmx)  
[Operator Fusion in XLA: Analysis and Evaluation (2023)](https://arxiv.org/pdf/2301.13062)  
[How Much Can We Gain From Tensor Kernel Fusion on GPUs? (2024)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10551817)

# Contest Organizers

Yarong Mu ([ymu@google.com](mailto:ymu@google.com)) & Weiren Yu ([weirenyu@google.com](mailto:weirenyu@google.com))

# Contest Sponsor

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/500px-Google_2015_logo.svg.png" width="120">
