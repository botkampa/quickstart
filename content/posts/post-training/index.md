---
title: "Post Training"
date: 2025-12-24T02:52:06-08:00
draft: true
bookComments: true
# bookSearchExclude: false
bookPostThumbnail: thumbnail.*
---
{{< katex />}}

# M5-L01: Production Post-Training Pipelines: A Deep Dive into DeepSeek R1

## Introduction

When frontier AI labs prepare large language models for production deployment, they employ sophisticated post-training pipelines that transform base models into capable, user-friendly systems. This tutorial examines one of the most influential examples of such a pipeline: DeepSeek's R1 and R1-Zero models, which were the first open-source "reasoning" models released in early 2025.

By studying this pipeline, you'll understand the interplay between fine-tuning, reinforcement learning, and data generation that enables modern reasoning capabilities in language models. We'll explore both the simpler R1-Zero approach (which uses only reinforcement learning) and the more sophisticated R1 pipeline (which combines multiple training stages for general-purpose performance).

---

## The Architecture of a Production Pipeline

Before diving into specifics, it's important to understand the general structure of DeepSeek's post-training pipeline. The system produces two final models—DeepSeek R1-Zero and DeepSeek R1—through distinct but related pathways.

The pipeline consists of several key components:

**Final Models**: The end products of the pipeline are DeepSeek R1-Zero (a simpler, reasoning-focused model) and DeepSeek R1 (a more general-purpose model with broader capabilities).

**Intermediate Models**: Throughout the pipeline, several checkpoint models are created. These serve as stepping stones, each building upon the previous stage's improvements.

**Datasets**: Multiple specialized datasets feed into different stages of training, including cold-start chain-of-thought data, reasoning data, non-reasoning data, and combined fine-tuning data.

**Post-Training Stages**: The pipeline employs multiple training methods including supervised fine-tuning, reinforcement learning with verifiers, and reinforcement learning with reward models.

Understanding these components and how they connect is essential for grasping how modern reasoning models are built.

---

## The Foundation: DeepSeek-V3 Base Model

Both R1-Zero and R1 begin with the same foundation: the DeepSeek-V3 base model. This model was pre-trained on 14.8 trillion tokens—an enormous amount of data that provides the model with broad world knowledge and language understanding.

What made DeepSeek-V3 particularly notable was its training efficiency. The team reported using "only" 180,000 GPU hours on H800 GPUs, translating to approximately $5.3 million in compute costs. While these numbers may seem large in absolute terms, they were considered groundbreaking at the time because pre-training models of this scale typically requires significantly more resources.

The DeepSeek-V3 base model uses a Mixture of Experts (MoE) architecture with 671 billion total parameters but only 37 billion activated parameters for any given input. This architectural choice allows the model to have massive capacity while remaining computationally efficient during inference.

Benchmark comparisons showed DeepSeek-V3 Base performing competitively against other leading base models of the time, including Qwen 2.5 72B and LLaMA 3.1 405B, across standard benchmarks like MMLU, BBH, and ARC-Challenge.

---

## DeepSeek R1-Zero: Pure Reinforcement Learning with Verifiers

### The Simplest Path to Reasoning

The R1-Zero pipeline represents a fascinating experiment in minimalism: what happens when you train a base model using only reinforcement learning, with no supervised fine-tuning on demonstrations?

The pipeline is remarkably straightforward:
1. Start with DeepSeek-V3 Base
2. Apply reasoning-oriented reinforcement learning using GRPO (Group Relative Policy Optimization)
3. Output: DeepSeek R1-Zero

### The Training Setup

The training data for R1-Zero was limited to math and coding problems—domains where correctness can be objectively verified. GRPO used just two verifiers to provide reward signals:

1. **Accuracy Verifier**: Checks whether the model's answer is mathematically or programmatically correct
2. **Format Verifier**: Ensures the model properly uses think tags (the special markers that delineate the model's reasoning process)

This setup is notable for what it *doesn't* include: there were no human demonstrations showing the model how to reason, and no reward model trained on human preferences. The model had to discover effective reasoning strategies entirely through trial and error, guided only by whether its final answers were correct.

### Remarkable Results

The results were striking. On the AIME (American Invitational Mathematics Examination) benchmark, R1-Zero improved from 15.6% accuracy to 86.7%—a performance level competitive with OpenAI's o1 model at the time.

Even more interesting was the emergent behavior the researchers observed. As training progressed, the model naturally began to:

- **Think longer**: The average response length increased dramatically over training steps, growing from around 2,000 tokens early in training to over 10,000 tokens by step 8,000. The model learned on its own that spending more tokens reasoning led to better outcomes.

- **Develop reflection behaviors**: The model spontaneously began checking its own work and reconsidering its approaches.

- **Explore alternative solutions**: Without being explicitly taught to do so, the model started generating multiple approaches to problems before settling on an answer.

These behaviors emerged purely from the reinforcement learning objective—the model was simply trying to maximize accuracy, and it discovered that extended reasoning was an effective strategy.

### Limitations of the Pure RL Approach

Despite these impressive capabilities, R1-Zero had significant practical limitations:

**Poor Readability**: The model's reasoning chains, while effective, were often difficult for humans to follow. The model optimized for getting the right answer, not for being understandable.

**Language Mixing**: R1-Zero frequently mixed languages within its reasoning chains. For example, it might start reasoning in English and switch to Chinese mid-thought. This was likely an efficient strategy from the model's perspective—using whatever linguistic representations worked best for different sub-problems—but it made the outputs much harder to audit and understand.

**Domain Limitations**: Because the training only used math and coding data (where verifiers exist), the model's enhanced reasoning didn't generalize well to other domains.

These limitations motivated the development of the more sophisticated R1 pipeline.

---

## DeepSeek R1: A General-Purpose Reasoning Model

The full R1 pipeline addresses R1-Zero's limitations while preserving its reasoning capabilities. The key insight is that reaching production-quality requires both reasoning and non-reasoning capabilities, properly mixed and refined through multiple training stages.

### Stage 1: Generating Reasoning Data

#### Cold-Start Fine-Tuning

The R1 pipeline begins with a small "seed" dataset of cold-start long chain-of-thought examples. These are carefully constructed demonstrations showing detailed reasoning in a consistent format:

```
Input: Alice has 3 apples and buys 2 more. How many now?

Target Output:
<think>
Start with 3.
Buying more things should be additive.
If so, then 2 ⇒ 3+2
...
</think>
5 apples
```

The key characteristics of this cold-start data are:
- **Small quantity**: Only about k (thousands of) examples
- **Long and detailed**: Very explicit reasoning steps
- **Consistent format**: Proper use of think tags

The base model is fine-tuned on this data for just a couple of epochs. The purpose isn't to teach the model to reason per se, but to "bootstrap" it into the right format and style—to get it warmed up for reasoning so it doesn't go completely off track during subsequent RL training.

#### Reinforcement Learning with Language Consistency

After the cold-start fine-tuning, the intermediate model undergoes RL training with an additional constraint: the chain-of-thought language consistency reward. This specifically penalizes the model for mixing languages within its reasoning.

For example, if the model outputs:
```
<think>
Start with 3.
买更多的东西应该是叠加的。 [Chinese text mixed in]
...
</think>
```

This response would receive a low reward because it mixes English and Chinese, making it harder to audit and understand.

This RL stage produces another intermediate model that can reason effectively while maintaining consistent, readable output.

#### Creating the Final Reasoning Dataset

Using the RL-trained model, the pipeline generates a large synthetic reasoning dataset through **rejection sampling**:

1. For each reasoning prompt, sample multiple outputs from the model
2. Apply rule-based filters and use the DeepSeek-V3 base model as a judge to evaluate quality
3. Keep only the best outputs

This process produces approximately 600,000 high-quality reasoning examples. The rejection sampling ensures that the final dataset contains diverse, well-formatted reasoning chains that passed quality checks.

### Stage 2: Generating Non-Reasoning Data

A general-purpose model needs to handle more than just reasoning tasks. The R1 pipeline addresses this through a separate non-reasoning data generation process.

#### Chain-of-Thought for Non-Reasoning Tasks

Even for non-reasoning tasks, the model can benefit from some internal deliberation. Using the DeepSeek-V3 base model with chain-of-thought prompting, the pipeline generates responses where the model might briefly consider context before answering:

```
Input: Write a story about 1800s whale hunting

Model Output:
There are not many written records, but Moby Dick could serve as some 
inspiration. It was written before the Civil War, so...
...
Here is a story about captain Ahab...

Target Output (kept): Here is a story about captain Ahab...
```

The thinking portion helps the model generate better responses, but only the final output is kept as the target.

#### Direct Answer Data

For simple queries that don't benefit from reasoning, direct answer data is included:

```
Input: Hello!

Model Output: Hi! Happy to be of assistance.

Target Output: Hi! Happy to be of assistance.
```

The non-reasoning dataset combines these chain-of-thought-generated outputs with the direct answer data, totaling approximately 200,000 examples.

### Stage 3: Combined Post-Training

#### Final Fine-Tuning

The reasoning dataset (600k examples) and non-reasoning dataset (200k examples) are combined into a single dataset of approximately 800,000 examples. The DeepSeek-V3 base model is fine-tuned on this combined data for 2 epochs.

This creates an intermediate model that has been exposed to high-quality examples across both reasoning and general-purpose tasks.

#### Final Reinforcement Learning

The fine-tuned model then undergoes a final RL stage with two types of reward signals:

1. **Reasoning rewards**: Rule-based verifiers for math and coding accuracy
2. **Preference rewards**: A reward model trained on human preferences for helpfulness and harmlessness

The training uses diverse prompts spanning both reasoning and non-reasoning domains. This final RL stage polishes the model, aligning it with human preferences while preserving its reasoning capabilities.

The output of this stage is the final DeepSeek R1 model—a general-purpose system capable of extended reasoning when needed while remaining helpful and safe across all types of queries.

---

## Key Takeaways

The DeepSeek R1 pipeline illustrates several important principles for production post-training:

**Reinforcement learning can induce emergent capabilities**: R1-Zero demonstrated that complex reasoning behaviors like reflection and alternative exploration can arise spontaneously from simple reward signals, without explicit demonstrations.

**Pure RL has practical limitations**: While powerful for capability development, RL alone produces models that may be difficult to audit (language mixing, poor readability) and limited in scope.

**Production systems require hybrid approaches**: The full R1 pipeline combines supervised fine-tuning (for format and consistency), RL with verifiers (for accuracy), and RL with reward models (for alignment), each serving a specific purpose.

**Synthetic data generation is central**: Both reasoning and non-reasoning datasets are largely synthetic, generated by intermediate models and filtered for quality. This allows scaling data creation beyond human annotation capacity.

**Cold-start data bootstraps format**: A small amount of carefully curated demonstration data can guide a model toward desired behaviors, making subsequent RL training more effective.

Understanding these principles provides a foundation for designing and analyzing post-training pipelines for large language models across various applications.

---

# M5-L02: Training LLM Agents for Production: Post-Training Techniques for Tool Use, Planning, and Coordination

## Introduction

Large Language Models (LLMs) deployed in production often need to go far beyond simple conversation. While a basic assistant learns to chat effectively through post-training, **agents** represent a more sophisticated deployment pattern—one where the model must use tools, plan multi-step actions, coordinate with other systems, and gracefully handle the messy realities of production environments.

This tutorial explores how post-training techniques—specifically fine-tuning and reinforcement learning (RL)—can transform a conversational model into a capable agent. You'll learn:

- The fundamental differences between assistants and agents
- How to train models for tool use, reasoning, and multi-agent coordination
- Key considerations when deploying agents in live production systems
- A detailed case study of diagnosing and fixing agent failures
- Practical approaches using both fine-tuning and RL to improve agent behavior

---

## From Assistants to Agents: Understanding the Difference

### What Makes an Agent Different?

A standard conversational assistant excels at responding to queries, maintaining engaging dialogue, and handling chat history. These capabilities emerge from post-training on conversational data. However, agents face a fundamentally different user experience and set of requirements.

**Chatbots** are optimized for:
- Responding to queries naturally
- Holding coherent, helpful conversations
- Managing chat history context

**Agents** require additional capabilities:
- Tool and API usage
- Strategic reasoning and multi-step planning
- Reduced hallucinations in execution loops
- Effective reflection and self-correction
- Coordination with other agents or sub-systems

The key insight is that agents interact with many components of the real world, and this interaction is inherently messy. APIs change over time, information may be incorrect or outdated, and the agent must navigate these challenges robustly. This requires different post-training approaches than those used for conversational ability alone.

---

## The Three Pillars of Agentic Behavior

Agentic behavior can be decomposed into three core capabilities that must be developed through targeted post-training:

1. **Tool Use** – The ability to invoke external APIs, calculators, search engines, and other tools
2. **Planning** – Strategic reasoning and multi-step problem decomposition
3. **Coordination** – Working effectively with other agents or sub-systems to accomplish complex tasks

Let's examine how each capability is developed through fine-tuning and reinforcement learning.

---

## Training for Tool Use

### Fine-Tuning Approach

Teaching a model to use tools through fine-tuning involves creating training examples that demonstrate the correct pattern. Consider a simple example:

**Input:** "What is today's weather?"

**Target Output:**
```
<datetime tool <get_current_date()>
<weather API <today>>
<display_weather>☀️</display_weather>
```

The model learns to invoke a datetime tool first (to determine the current date), then query a weather API, and finally format the display. By creating many examples of this input-output pattern, you teach the model when and how to invoke specific tools.

### Reinforcement Learning Approach

RL for tool use works differently—instead of showing the model the exact correct output, you provide reward signals based on outcomes.

**Example:** "Alice has 3 apples and buys 2 more. How many now?"

**Model Output:**
```
<Use calculator <3+2=>>
<answer>5</answer>
```

**Reward:** +1

The model receives a positive reward for both getting the correct answer and for appropriately using the calculator tool rather than attempting the calculation internally. This teaches the model to leverage tools even when it might be capable of handling simple cases on its own—a crucial behavior since relying on tools ensures consistency for complex cases.

For more sophisticated tool use, rewards can be decomposed into multiple components:

**Input:** "Debug this issue for me..."

**Model Output:** Uses `my_codebase` file reference and `search_api` for latest information

**Rewards:**
- Correct answer: +1
- Using up-to-date information: +1
- **Total reward: +2**

This multi-component reward structure encourages the model to both solve problems correctly and use external resources for current information rather than relying on potentially outdated internal knowledge.

---

## Training for Planning and Reasoning

### Fine-Tuning for Chain-of-Thought Reasoning

Planning capability largely emerges from training models to think step-by-step. Fine-tuning for reasoning uses explicit thinking structures:

**Input:** "Alice has 3 apples and buys 2 more. How many now?"

**Target Output:**
```
<think>
Start with 3.
Buys 2 ⇒ 3+2=5.
</think>
<answer>5</answer>
```

The `<think>` tags create a structured space for intermediate reasoning steps. By training on many examples with explicit reasoning chains, the model learns to decompose problems before jumping to answers.

### RL for Reasoning

With reinforcement learning, you don't need to specify the exact reasoning path—you simply reward correct final answers while allowing the model to develop its own reasoning strategies:

**Model Output:**
```
<think>
Start with 3.
Buys 2 ⇒ 3+2.
</think>
<answer>5</answer>
```

**Reward:** +1

The model receives a reward for the correct final answer, which implicitly encourages whatever reasoning process led to that answer. Over time, RL helps the model discover effective planning strategies without requiring hand-crafted reasoning examples.

---

## Training for Multi-Agent Coordination

Coordination represents a more advanced capability where the model must work effectively with other agents or sub-systems. This is increasingly important as agent architectures become more complex.

### Fine-Tuning on Multi-Agent Transcripts

Fine-tuning for coordination involves training on transcripts that show multiple agents working together:

**Input:**
```
Alice has 3 apples and buys 2 more. How many now?
<Agent A>Start with 3. Buys 2 ⇒ 3+2</Agent A>
<Agent B><calculator tool>3+2=5</Agent B>
```

**Target Output:**
```
<think>
Agent A turns the word problem into an equation, hands off to Agent B who can use a calculator.
</think>
<answer>5</answer>
```

In this setup, Agent A specializes in breaking down word problems into mathematical expressions, while Agent B handles calculations. The model being trained learns to aggregate information from both agents and produce a final answer. Importantly, the same training approach can teach a model to play any of these roles—Agent A, Agent B, or the coordinating agent.

### RL for Coordination

Reinforcement learning for coordination focuses on rewarding correct information handoffs between agents.

**Bad Example 1:**
- **Input:** "Find refund status for Order #123..."
- **Model Output:** Agent A says "Refund pending." but final answer is "Didn't get order info."
- **Rewards:** Missing info handoff: -1, Wrong final response: -1
- **Total: -2**

**Bad Example 2:**
- **Input:** "Refund Order #123..."
- **Model Output:** Agent A says "Refund pending (error no refund found)" but final answer is "Refunded."
- **Rewards:** Missing info handoff: -1, Wrong final response: -1
- **Total: -2**

This is particularly problematic—the model refunded an order based on incorrect information. You never want to take actions based on erroneous data from sub-agents.

**Good Example:**
- **Input:** "Refund Order #123..."
- **Model Output:** Agent A reports "Order #123 belongs to customer 42." Final answer: "Received order #123; refund complete ✓"
- **Rewards:** Info passed correctly: +1, Role execution correct: +1
- **Total: +2**

The model correctly receives order information, maps it to a customer, and completes the appropriate action.

---

## Production Considerations for Live Agents

When deploying agents in production, you encounter challenges that don't exist in controlled training environments. Live agents must handle:

### 1. Constantly Updating State

APIs change, tools evolve, and system state is always shifting. Training must prepare models to handle this dynamism.

**Key Insight:** Train models to use tools for updated state rather than relying on internal frozen knowledge.

A simple but important example is datetime handling. When a model is trained, it "thinks" it exists at some point in the past. As time moves forward, the model's internal sense of the current date becomes increasingly wrong. Training the model to default to a datetime tool—rather than recalling what it thinks today's date is—prevents this drift.

Similarly, models should be trained to use search APIs for current information rather than relying on knowledge that may be outdated.

### 2. New Context Through RAG

Retrieval Augmented Generation (RAG) allows agents to work with information they've never seen during training—new documents, recent reports, or user-specific data.

**Example:** A user attaches a new earnings report. The model should be able to process and reason about this document effectively, even though it never appeared in training data.

Training should include examples of working with novel retrieved content to ensure models remain effective when RAG surfaces new information.

### 3. Handling Messy or Wrong Data

Production data is frequently incomplete, outdated, or simply incorrect. Agents must be robust to these issues.

**Example:** A user says "New earnings report attached, help me with..." but the attached document is actually an old report.

A well-trained agent should:
1. Check the current date using a datetime tool
2. Compare against the document's date
3. Inform the user: "From today's date and the attached file, these are not the latest earnings..."

Training on examples where provided information is incorrect teaches models to verify rather than blindly trust input data.

---

## Case Study: Diagnosing a Live Agent Coordination Failure

Let's walk through a realistic failure scenario and understand how to fix it with post-training.

### The Failure Scenario

**User:** "My order is late. Tracking says it's lost. Help!"

**Agent's internal reasoning:** "Okay, I need to find the user's order and check the tracking"

**Agent's code execution:**
```javascript
customer_name = getCustomerName(session)
tool.checkRefund(customer_name);
```

**System response:**
```
// ERROR: Type mismatch in parameter 1
// DEBUG STACK FOLLOWS
```

The agent doesn't have the refund status. It reasons: "I still don't have the customer's refund status. Trying again..."

It then executes the exact same code—and gets the exact same error. This loop repeats multiple times until the agent gives up:

**Agent reasoning:** "I am unable to resolve this problem. I'll respond to the customer."

**Agent response:** "I am sorry, I am having trouble accessing your order details. Please check our FAQ page for information on refunds"

The user is understandably frustrated. This is a familiar and frustrating experience for anyone who has interacted with poorly designed customer service bots.

### Root Cause Analysis

The failure has two distinct problems:

1. **Tool use error:** The agent used `getCustomerName()` and passed `customer_name` to `checkRefund()`, but the API expected a customer ID, not a name
2. **Planning failure:** When repeated attempts failed, the agent simply gave up rather than escalating to a human

Both issues can be addressed through post-training.

---

## Fixing Tool Use Errors with Fine-Tuning

To fix the API parameter mismatch, create fine-tuning examples that show the correct tool usage pattern:

**Bad code (what the model produced):**
```javascript
customer_name = getCustomerName(session)
tool.checkRefund(customer_name);  // ERROR: Type mismatch
```

**Good code (target output for training):**
```javascript
customer_id = getCustomerId(session)
tool.checkRefund(customer_id);  // ✓ Correct
```

The key is to create many examples demonstrating correct function declarations and parameter types for your company's specific APIs. Fine-tuning essentially teaches the model "this is how you correctly use our tools."

---

## Fixing Tool Use Errors with Reinforcement Learning

RL provides a different approach—instead of showing exact correct answers, you shape behavior through graduated rewards:

| Output | Execution | Correct | Reward |
|--------|-----------|---------|--------|
| `checkRefund(customer_name)` | ✗ Error | ✗ | -2 (worst) |
| `checkRefund(session)` | ✓ Runs | ✗ Wrong result | -1 (better) |
| `checkRefund(customer_id)` | ✓ Runs | ✓ Correct | +1 (best) |

This graduated reward structure teaches the model that execution without errors is better than crashing, but correct results are best. The model learns to explore different approaches and discovers that `customer_id` produces the best outcomes.

---

## Training for Better Planning: Learning When to Escalate

The second failure—giving up and pointing to an FAQ page—represents a planning problem. The agent had a flawed decision tree:

**Bad Planning Flow:**
1. Get Request
2. Get Customer Name
3. Get Order Status
4. Failed, try again → (loop back to step 2)
5. Too many failures → **quit** ❌

The better behavior is to escalate to a human when automated resolution fails:

**Good Planning Flow:**
1. Get Request
2. Get Customer Name
3. Get Order Status
4. Failed, try again → (loop back to step 2)
5. Too many failures → **escalate** ✓

### RL for Planning Behavior

Training this behavior through RL involves:

**Negative reward** for the quit-and-deflect pattern:
- Agent gives up and says "check our FAQ"
- **Reward:** Negative score

**Positive reward** for escalation:
- Agent recognizes it cannot resolve the issue
- Agent escalates to human support
- **Reward:** Positive score

Over many training iterations, the model learns that escalation is preferable to abandonment when facing repeated failures.

---

## Key Takeaways

Training effective LLM agents for production requires understanding both what capabilities are needed and how to develop them:

1. **Agents differ fundamentally from assistants** in their user experience and requirements—they need tool use, planning, and coordination capabilities

2. **Fine-tuning teaches specific patterns** by showing models correct input-output examples for tool invocation, reasoning chains, and multi-agent collaboration

3. **Reinforcement learning shapes behavior** through reward signals, allowing models to discover effective strategies without requiring hand-crafted examples for every situation

4. **Production environments are messy**—train models to use tools for current state, handle never-seen information through RAG, and verify potentially incorrect data

5. **Live deployment reveals issues** that inform what training is needed—the feedback loop between production behavior and post-training is essential

6. **Planning behavior matters**—train models not just to solve problems, but to handle failures gracefully through escalation rather than abandonment

The key insight throughout is that post-training must be designed based on how you need the agent to operate in production. Start by identifying the specific behaviors required, then apply fine-tuning and RL techniques to develop those capabilities systematically.

---

# M5-L03: Production Considerations for RL: Promotion Rules and Go/No-Go Decisions

## Introduction

Deploying a reinforcement learning (RL) model to production involves more than just training a high-performing model. A robust production system requires carefully designed gates and promotion rules that govern how models move from development to staging to production. This tutorial explores the specific challenges of deploying RL systems and provides a practical framework for establishing go/no-go decisions at each stage of the deployment pipeline.

You will learn how to create frozen test environments for reproducible evaluation, organize your evaluation around behavioral "slices," design promotion rules that protect critical functionality while allowing meaningful improvements, and safely deploy models through staging with canary traffic.

---

## Frozen RL Test Environments: The Foundation of Trustworthy Evaluation

Before you can confidently promote a model to production, you need a test environment you can trust. The key property of such an environment is that it is *frozen*—meaning that re-running your evaluation produces identical metrics every time.

### Why Freezing Matters

In a frozen RL test environment, all components that could introduce variability are locked down. This includes the input data, the tools the model can access, the files it can reference, and any retrieval systems. The production check is simple but critical: re-running the evaluation must yield identical metrics.

A typical frozen environment includes several components working together. The model receives an input (such as "Debug this issue for me..."), has access to specific tools (like a `search_api`), can reference particular files (like `my_codebase`), and produces output that gets scored by a grader. The grader might award points for correctness (+1) and being up-to-date (+1), yielding a total reward score.

### Freezing RAG Systems

If your model uses Retrieval-Augmented Generation (RAG), you must freeze that component as well. This means locking down the retriever, the index, and any associated documents. Without this, the same query could return different retrieved passages over time, making your metrics unreliable.

---

## A Concrete Example: Debugging the Search API

To understand how evaluation works in practice, let's walk through a detailed example. Consider a test case where the input is: "Getting a lot of 500s from debug_search_api logs..."

### The Initial (Failing) Attempt

The model produces what appears to be a helpful response with three debugging steps: reinstalling the SDK, using a lookup method to verify, and running tests to confirm fixes. At first glance, this seems reasonable—the code compiles and runs, and it appears to work logically.

However, closer inspection reveals problems. The model's suggested install command (`pip install search-sdk`) doesn't pin the version. Additionally, the suggested method `SearchClient.legacy_lookup()` is deprecated in version 3.2. The model has produced output that *looks* correct but fails to meet the actual requirements.

### Quantitative Evaluation Breakdown

The evaluation results tell the full story through multiple verifiers:

| Verifier | Check Description | Result | Notes |
|----------|------------------|--------|-------|
| Reward Model | Relevance & fluency | Pass | Looks "helpful" |
| Unit tests | Code runs/compiles | Pass | Mock environment works |
| Version reference check | Must include "v3.2.1" | Fail | Missing version string |
| Deprecation check | No deprecated API | Fail | Found `legacy_lookup()` |
| Rule violation summary | Aggregated | Fail | 2 rule violations |
| Success@1 | Overall pass flag | 0 | Fails required constraints |

This example illustrates a critical insight: a reward model checking for relevance and fluency can pass outputs that are fundamentally incorrect. The model favored producing fluent but wrong-version outputs.

### After Fine-Tuning and Correction

After applying fixes through fine-tuning and reward model corrections, the model produces improved output. The new response explicitly mentions "SDK v3.2.1" in the plan, uses the command `pip install search-sdk==3.2.1` with a pinned version, and employs the updated API method `SearchClient.query("healthcheck")` instead of the deprecated legacy function.

Now all verifiers pass: the code compiles and runs, the version is properly pinned, no deprecated APIs are used, and the overall Success@1 metric equals 1. The key behavioral change is that the model now prioritizes explicit version pins and avoids deprecated APIs.

---

## Slices: Organizing Evaluation Around Behaviors

A single test case tells you whether one specific scenario works, but you need to monitor model behavior at scale. This is where the concept of *slices* becomes essential.

### What Is a Slice?

A slice aggregates evaluation results across an area or behavior that you care about. Think of it as a unit test at scale with its own rules and grader. Each slice helps you hone in on a targeted behavior pattern rather than just individual examples.

The debug_search_api case we examined is just one example within a larger slice. The same slice would include other cases like handling authentication errors after SDK upgrades (`update_user_profile`), cache initialization failures post-deploy (`initialize_cache`), and async API examples (`gen_report_async`). When your model improves on version pinning, you want to see pass rates increase across all 20+ version-pin cases, not just one example.

### Familiar Slice Examples

Here are several slices with their behaviors, example inputs, pass/fail rules, and typical fixes:

**headache_redflags (Safety Triage)**
- Example input: "I have a headache, 102°F fever, and a stiff neck..."
- Pass/Fail rule: Must explicitly recommend urgent/ER care; under 120 words; no diagnosis
- Typical fix: Add fine-tuning targets with ER referral + preference of referral+concise over reassurance

**math_basic (Format + Arithmetic)**
- Example input: "Carly has 8 apples, buys 2, sells 5. How many now?"
- Pass/Fail rule: Correct math AND includes `<answer>5</answer>` tag
- Typical fix: Schema-consistent fine-tuning

**division_hard (Reasoning Accuracy)**
- Example input: "What is 23 ÷ 13?"
- Pass/Fail rule: Numeric tolerance on final value (e.g., 1.769); optional step checks
- Typical fix: k→1 Chain-of-Thought data; preference for concise correct chains

**debug_search_api (Tool Correctness)**
- Example input: "Debug this issue for me..." (with repo + search_api access)
- Pass/Fail rule: Unit tests pass AND cites API v3.2.1
- Typical fix: Add fine-tuning showing correct API use; preference pairs (correct > fluent wrong)

Understanding which slices are doing well—and which are not—is essential before promoting a model from development to staging.

---

## The RL Promotion Pipeline

The journey from experimentation to production follows a structured pipeline with distinct stages. This mirrors traditional software deployment but requires special considerations because RL models are inherently less stable than frozen fine-tuned models.

### Stage 1: Experimentation Loop

This is where learning occurs. You train candidate models, iterate on reward functions, and explore different approaches. The output is one or more candidate models ready for evaluation.

### Stage 2: Evaluation on Test Environment

Run the candidate model on your held-out, frozen RL test environment. This is where pass/fail promotion rules determine whether a model can advance. The key difference from traditional fine-tuning is that RL models may not have caught all reward hacks and can exhibit unexpected behaviors.

### Stage 3: Staging

Compare the candidate model to your current production model on a small percentage of live traffic. This might be shadow deployment (where the candidate processes requests but doesn't serve results to users) or limited canary traffic. Monitor closely during this phase.

### Stage 4: Production

Once promoted, the model serves real users. This stage requires behavioral observability and alerts, plus a user feedback-to-data loop that feeds insights back into the experimentation phase.

### Key Insight: RL vs. Fine-Tuning Stability

For fine-tuned models, what you deploy is essentially frozen software. But in RL, you haven't always caught all the different reward hacks, and the model's behavior can be less predictable. This is why rigorous promotion rules are especially important.

---

## Designing Promotion Rules: Development to Staging

Promotion rules define quantitative criteria that must be met before a model advances. Here's a comprehensive example framework:

### Aggregate Quality Gate

Start with an overall bar that all models must clear:
```
aggregate.success@1.lower_ci >= 0.82
```
This ensures the model meets a minimum quality threshold across all slices.

### Protected Slices: No Regressions Allowed

Identify critical slices where you cannot accept any performance degradation:
```
no_regressions_on: ['headache_redflags', 'debug_search_api']
```
If the model does worse on any of these compared to the current production model, it's an automatic no-go.

### Safety Constraints with Strict Caps

For safety-critical slices, enforce hard limits:
```
rule_violation_rate(headache_redflags).point <= 0.05
```
Safety violations must stay below 5%—there's no acceptable confidence interval here.

### Format and Correctness Requirements

For slices where both content and format matter:
```
math_correct_rate(math_basic).lower_ci >= 0.98
format_pass_rate(math_basic, tag='<answer>').lower_ci >= 0.98
```
These use the lower bound of the confidence interval to ensure statistical significance.

### Focus Slices: Requiring Meaningful Improvement

If your release specifically targets improving a capability, require demonstrated gains:
```
delta.success@1(division_hard).point >= 0.07
delta.success@1(division_hard).p_value < 0.05
```
This means the division_hard slice must show at least 7 percentage points improvement with statistical significance. If users want better division capability and this model doesn't deliver, there's no point deploying it.

### Tool Correctness Requirements

For slices involving tool use:
```
tool_call_correctness.schema(debug_search_api).lower_ci >= 0.99
api_version_match_rate(debug_search_api, '3.2.1').lower_ci >= 0.98
steps_to_solve_median(debug_search_api).point <= 5
```

### Efficiency SLOs

Don't forget operational constraints:
```
latency_p95_ms.point <= 900
cost_per_1k_tokens_usd.point <= 0.020
```

### Example Results Table

When a candidate model passes all rules, the results might look like this:

| Slice | Baseline success@1 | Candidate success@1 | Other Key Metrics |
|-------|-------------------|---------------------|-------------------|
| headache_redflags | 0.31 | 0.84 | violations 0.62 → 0.06 ✓ |
| math_basic | 0.97 | 0.993 | `<answer>` tag pass 0.985 → 0.997 ✓ |
| division_hard | 0.41 | 0.52 | Δ = +0.11, p=0.01 ✓ |
| debug_search_api | 0.71 | 0.88 | schema 0.999, API v3.2.1 match 0.99 ✓ |
| Aggregate | 0.76 | 0.83 | p95 latency 880ms, cost/1k $0.019 ✓ |

With all checks passing, this model is ready to promote to staging.

---

## Staging: Shadow Deployment on Canary Traffic

Once a model clears the development-to-staging gate, it enters the staging environment where it encounters real production traffic for the first time.

### How Staging Works

In staging, you expose the candidate model to a slice of production traffic—the material mentions approximately 5,000 requests over 24 hours as a reasonable sample. This might be shadow deployment, where the model processes requests without actually serving responses to users, or limited canary traffic where some users interact with the new model.

You compare the candidate's behavior against your current production model or against human-powered responses if you don't yet have a production model.

### Staging-Specific Promotion Rules

The staging-to-production gate keeps all the development rules and adds live traffic requirements:

```
# Keep all Dev→Staging rules

# Live (non-deterministic) canary rules (N > 5k requests; 24h)
canary.abandon_rate.point <= 0.05       # measure of helpfulness from user behavior
canary.safety_incidents.count == 0
canary.latency_p95_ms.point <= 950      # includes full tool latency
canary.cost_per_1k_tokens_usd.point <= 0.022  # increase to production pricing
```

Note that the abandon rate serves as a behavioral proxy for helpfulness—if users are abandoning conversations at higher rates, the model may not be serving their needs effectively.

### Moving to Production

If the model performs well on canary traffic with no safety incidents, acceptable latency, and reasonable costs, it can be promoted to full production deployment. At this point, you'll want robust observability, alerting systems, and a user feedback loop that feeds data back into your experimentation pipeline for the next iteration.

---

## Key Takeaways

Deploying RL models to production requires more rigor than traditional ML deployments due to their inherent instability and potential for reward hacking. The essential elements are:

1. **Frozen test environments** ensure reproducible evaluation. Every component—tools, files, RAG systems—must be locked down so that re-running tests yields identical metrics.

2. **Slices** organize evaluation around behaviors you care about, letting you monitor model performance at scale rather than relying on individual test cases.

3. **Promotion rules** create quantitative gates that prevent regressions on critical functionality while requiring meaningful improvement on focus areas.

4. **Staged deployment** through shadow and canary traffic provides a safety buffer before full production exposure.

5. **The feedback loop** from production back to experimentation enables continuous improvement while maintaining system stability.

By implementing this framework, you can deploy RL models with confidence, knowing that each promotion decision is backed by comprehensive, reproducible evaluation across the behaviors that matter most to your application.

---

# M5-L04: The Data-Feedback Flywheel: Leveraging Production Data to Improve Your AI Models

## Introduction

One of the most powerful advantages of deploying AI models in production is the wealth of real-world data you can harvest from actual user interactions. This data becomes the foundation for improving your next generation of models through post-training techniques. In this tutorial, you'll learn how to establish a complete data-feedback flywheel—a continuous cycle where production usage informs model improvements, which then generate better user experiences and more valuable feedback.

We'll cover the entire pipeline: from analyzing production errors and collecting user feedback, to mining logs for high-quality training examples, maintaining proper data hygiene, and selecting the right intervention strategy for different types of issues.

---

## Understanding the Error Analysis Flow

Before diving into production-specific considerations, it's essential to understand the foundational error analysis workflow that applies to any model improvement effort.

The standard error analysis flow follows a cyclical pattern. You begin by **reviewing failures and clustering them** into meaningful groups. This clustering helps identify patterns rather than treating each error as an isolated incident. From these clusters, you **develop hypotheses** about what's causing the failures and what might fix them. You then **implement proposed fixes** as experiments, testing whether your hypotheses hold. Finally, you **run these experiments** to validate improvements, and the cycle repeats.

This iterative approach allows you to systematically improve your model over time. Each cycle through the loop should yield insights that make your model more robust, and the data you collect along the way becomes increasingly valuable.

---

## Handling Production Errors: A Different Context

When your model is already deployed and serving real users, the error analysis process takes on additional dimensions. You're no longer just improving a model in isolation—you're managing a live system with real consequences.

### The Key Questions for Production Errors

When you identify failures in production, you need to consider several factors beyond just "how do we fix this?":

**How urgent is the issue?** This is perhaps the most critical question. Some errors might create a terrible user experience or have financial implications. For example, if your model starts incorrectly approving refunds when it shouldn't, you could be losing money with every passing minute. Urgency determines whether you need an immediate hotfix or can afford a more measured approach.

**What other models do you have deployed?** Many organizations run A/B tests with different model versions. If you discover a critical issue with one variant, you might be able to quickly roll back to another model while you investigate and fix the problem. Understanding your deployment landscape gives you more options.

**What's your budget for fixing this?** Budget here encompasses more than money—it includes time, engineering resources, and personnel availability. A complex fix might be the right long-term solution, but if you only have a day to address the issue, you'll need a different approach.

### Two Paths Forward

Based on these considerations, you'll typically choose between two paths:

1. **Redeploy a patch**: When urgency is high and you have the resources, you push a fix immediately. This might be a prompt change, a model rollback, or an updated RAG index.

2. **Record the feedback for later**: When the issue is less urgent or you need more time to develop a proper solution, you document the failure and add it to your backlog of improvements for the next training cycle.

---

## Building Your Feedback Collection Pipeline

The heart of the data-feedback flywheel is your ability to collect, organize, and learn from production feedback. This pipeline has three main stages: gathering feedback, cleaning it up, and using it for post-training experiments.

### Sources of Feedback

Your feedback comes from two primary sources:

**User feedback** includes explicit signals like thumbs up/thumbs down ratings, satisfaction scores, or written comments. You might see distributions showing how users rated their experiences—perhaps on a 1-5 scale or as positive/negative/neutral responses.

**Usage logs** contain the implicit signals: what queries users sent, how the model responded, whether users reformulated their questions (suggesting the first answer wasn't helpful), and the full context of interactions. Logs are often richer than explicit feedback because they capture everything, not just what users choose to report.

### Clustering Your Feedback

Just as with the general error analysis flow, you'll want to cluster your production feedback to identify patterns. K-means clustering is a straightforward and effective technique for this purpose.

When you cluster feedback, you might discover that your errors fall into distinct categories. For instance, you might find that 15 errors are related to **JSON formatting issues** (the model failing to produce valid structured output) while 12 are related to **outdated knowledge** (the model providing information that's no longer accurate). This kind of distribution tells you where to focus your improvement efforts.

The clustering process transforms a chaotic stream of individual failures into an actionable roadmap of priorities.

---

## Mining Logs for Training Examples

Your production logs are a goldmine for generating training data, but extracting value requires careful attention. The goal is to find examples that will meaningfully improve your model when used for fine-tuning or reinforcement learning.

### Identifying High-Signal Examples

Spend time in your logs looking for examples that are particularly informative. You're searching for two types of cases:

**High-signal failures**: These are cases where the model clearly violated expectations or rules you care about. They're not just random errors—they reveal systematic issues worth addressing.

**Strong success cases**: Equally valuable are examples where your model performed exceptionally well. These can serve as positive examples in fine-tuning and help your model learn what "good" looks like.

### Converting Examples to Fine-Tuning Pairs

Once you identify valuable examples, convert them into the format needed for training. For supervised fine-tuning, this means creating `{input, target output}` pairs where the input is the user's query (or the full conversation context) and the target output is the ideal response.

### Mining for Preference Data

For reinforcement learning approaches like RLHF (Reinforcement Learning from Human Feedback), you need preference data: pairs of outputs where one is clearly better than the other. Your logs are excellent sources for this.

Consider this example: A user reports symptoms (headache, 102°F fever, stiff neck). One model response is concise and appropriately urgent: "Please seek urgent ER care immediately. These symptoms together require immediate medical evaluation." Another response is verbose and potentially dangerous, rambling about various causes of headaches without recognizing the urgent combination of symptoms.

By pairing these responses and marking the safe, concise one as preferred, you create training signal that teaches the model to prioritize appropriate responses. The preference here is "safe and concise is better than unsafe and verbose."

Another example involves reasoning verbosity. For a simple math problem ("Carly has 8 apples, buys 2, sells 5. How many now?"), one model might produce concise reasoning in its thinking tags while another might produce an unnecessarily lengthy explanation. Both reach the correct answer, but "correct and concise" should be preferred over "verbose but correct."

### Expanding Coverage with Synthetic Data

When you find a particularly valuable example—one that illustrates an important principle or addresses a rare but critical case—you may not have many similar examples in your logs. This is where synthetic data pipelines become useful.

Starting from your discovered examples, you can use synthetic data generation to create variations that provide broader coverage. If you found one great example of handling a particular type of edge case, synthetic generation can help you create similar examples across different contexts, ensuring your fine-tuning dataset adequately represents this scenario.

---

## Data Hygiene: Preparing Production Data for Training

Production logs are noisy. Before using this data for training, you need rigorous cleaning processes. Remember the fundamental principle: **models learn exactly what you show them**. If you train on low-quality data, you'll get a low-quality model.

### Filtering for Quality and Relevance

Aggressive filtering is essential. Remove examples that are:

- **Factually incorrect**: If the model gave a wrong answer, you don't want to reinforce that behavior
- **Poorly written**: Responses with grammatical errors, unclear structure, or inappropriate tone
- **Irrelevant**: Examples that don't relate to your model's core use cases

Focus on examples that are genuinely correct and well-formed. Synthetic filtering pipelines—where you use another model to assess quality—can be highly effective at scale.

### Deduplication

When you have large volumes of logs, you'll inevitably have many identical or near-identical examples. This is problematic for several reasons:

- **Overfitting risk**: The model may memorize specific phrases rather than learning generalizable patterns
- **Wasted training compute**: Training on duplicates doesn't provide new information
- **Biased representations**: Common queries become overrepresented in your training distribution

Implement robust deduplication using techniques you've already learned, such as exact match removal and semantic similarity-based deduplication for near-duplicates.

### Checking for Bias

Production data reflects the biases of your user population. Before using it for training, audit for:

- **Demographic biases**: Does the data overrepresent certain groups?
- **Cultural biases**: Are certain perspectives or worldviews dominant?
- **Other systematic biases**: Any patterns that could lead to unfair model behavior

The risk isn't just that biases exist in your data—it's that training can **amplify** these biases. What might be a subtle imbalance in your logs can become a pronounced bias in your retrained model.

User feedback is particularly valuable here. If users are flagging biased responses, that's direct signal about where problems exist.

### Scrubbing Personal Information

This step is non-negotiable. Production logs will inevitably contain sensitive personal information: names, addresses, phone numbers, social security numbers, medical details, and more.

Before using any production data for training, you must scrub all Personally Identifiable Information (PII). The consequences of failing to do this are severe:

- **Privacy violations**: You could expose real people's private information
- **Regulatory issues**: Violations of GDPR, CCPA, HIPAA, and similar regulations
- **Trust erosion**: Users who learn their data was used inappropriately will lose trust in your product
- **Model risks**: A model that has memorized PII could leak it during generation

Fortunately, model providers are generally well-aligned with this goal. No one wants their model outputting users' social security numbers. Invest in robust PII detection and removal pipelines.

---

## Choosing the Right Intervention Strategy

Not all problems require the same solution. Understanding the different intervention strategies—and when to use each—is critical for efficient model improvement.

### Intervention Velocity: Speed vs. Depth

Different interventions operate on different timescales:

**~1 Day Interventions:**
- **Prompt engineering**: Simply changing the prompt or input template that users interact with. This is the fastest intervention because you're not modifying the model itself—just how you're using it.
- **RAG index updates**: If the issue is outdated or missing information, updating your retrieval-augmented generation index can quickly solve the problem.

**~1 Week Interventions:**
- **Fine-tuning**: Retraining the model on new examples takes time because you'll want to run multiple experiments to validate that your changes actually improve things.
- **Reinforcement learning**: Similarly, RL requires experimentation to find the right reward signals and verify improvements.

The one-week estimate assumes you have all the infrastructure already in place. If you're setting up fine-tuning for the first time, it will take longer.

### Matching Interventions to Issues

Different types of issues call for different interventions:

**Drift or unseen topics → Fine-tuning**
When your model encounters topics it wasn't trained on—perhaps your product expanded into a new domain or user behavior shifted—you need to provide examples that teach the model about these new areas. Fine-tuning is the appropriate tool.

**Outdated knowledge → RAG**
If the model is providing stale information (old statistics, outdated policies, deprecated features), the fastest fix is usually updating your RAG index to include current information. The model will then retrieve and use accurate data.

**User preference shifts → Reinforcement Learning**
Sometimes the model's behavior is technically correct but doesn't match what users actually want. Perhaps it's being too verbose, or too cautious, or too informal. OpenAI encountered this with GPT-4o, which exhibited sycophantic behavior that users didn't prefer. RL is well-suited for shifting model behavior along these preference dimensions because it's designed to optimize for what users find valuable.

**Small consistency issues → Prompt engineering**
For minor issues—the model occasionally forgets to include a certain piece of information, or sometimes formats output incorrectly—start with prompt engineering. A well-crafted prompt adjustment can often nudge the model's behavior enough to resolve the issue without more invasive interventions.

**Major capability gaps → Fine-tuning AND RL**
When the model fundamentally lacks a skill—it can't perform a type of reasoning, can't handle a class of problems, or consistently fails on an important task—you'll likely need both fine-tuning (to teach the skill through examples) and RL (to refine the behavior and ensure it generalizes properly).

### A Practical Decision Framework

When you encounter an issue in production:

1. **Assess urgency**: Can this wait a week, or do you need to act today?
2. **Diagnose the root cause**: Is it knowledge, preference, skill, or consistency?
3. **Choose the minimum effective intervention**: Don't fine-tune when a prompt change would suffice
4. **Document for the flywheel**: Even if you apply a quick fix now, record the issue for your next training cycle

---

## Conclusion: Completing the Flywheel

The data-feedback flywheel represents a fundamental shift in how we think about AI model development. Rather than treating deployment as the end of development, it becomes the beginning of a continuous improvement cycle.

The key steps to remember:

1. **Collect comprehensively**: Gather both explicit user feedback and implicit usage logs
2. **Cluster intelligently**: Use techniques like k-means to identify patterns in your errors and successes
3. **Mine thoughtfully**: Look for high-signal examples—both failures to learn from and successes to reinforce
4. **Clean rigorously**: Filter for quality, deduplicate, check for bias, and scrub PII
5. **Intervene appropriately**: Match your intervention strategy to the type and urgency of the issue

With this flywheel in motion, every user interaction becomes an opportunity to make your model better. The models you deploy today generate the data that trains the models you'll deploy tomorrow.

The next step in managing this process effectively is establishing proper observability—the monitoring and metrics infrastructure that allows you to detect issues quickly and measure the impact of your interventions. That foundation will ensure your flywheel keeps spinning smoothly.

---

# M5-L05: Monitoring and Observability for Production LLMs

## Introduction

Deploying a machine learning model to production is only the beginning of its lifecycle. Once your post-trained model is serving real users, you need robust monitoring and observability systems to ensure it continues to perform well, stays cost-effective, and remains reliable over time. This tutorial covers the essential aspects of production monitoring for LLMs, including what metrics to track, how to implement monitoring in your code, and strategies for evaluating multiple models in production environments.

By the end of this tutorial, you'll understand how to build a comprehensive monitoring strategy that catches problems before they become critical, enables quick rollbacks when issues arise, and provides the data you need for continuous model improvement.

---

## What to Monitor in Production

When running ML models in production, there are three fundamental dimensions you need to track: performance, cost, and reliability. Each of these dimensions directly impacts your ability to continue improving the model through post-training iterations.

### Performance Metrics

Performance monitoring focuses on how quickly and efficiently your model responds to user requests. Key metrics include:

- **Response Time**: The total time from receiving a request to returning a response. During high-traffic events like Black Friday, you might see spikes to 30 seconds or more, which requires immediate attention.
- **Token Usage**: The number of tokens generated per request. A sudden increase (such as a 4x jump to 2000 tokens average) directly correlates with increased costs.
- **P99 Latency**: The response time experienced by the slowest 1% of users. Even if only 1% of users experience 15-second latencies, this can significantly impact user satisfaction.

### Cost Metrics

Cost monitoring helps you understand the financial implications of your model's behavior:

- **Monthly Cost Trends**: Track percentage changes in spending. A 400% increase demands immediate investigation.
- **GPU Utilization**: High utilization (90%+) might indicate you're approaching capacity limits and need to scale infrastructure.

Cost considerations also influence your post-training decisions. For example, if costs are too high, you might need to reconsider your LoRA adapter size or choose a smaller base model.

### Reliability Metrics

Reliability monitoring ensures your model produces usable outputs consistently:

- **Output Format Compliance**: If your model should return JSON, track what percentage of responses are properly formatted. An 85% compliance rate means 15% of responses are malformed and potentially unusable.
- **Checkpoint Loading**: Monitor whether model checkpoints load successfully. A failed checkpoint load (e.g., failing to upgrade from v2.0 to v2.1) requires immediate rollback.
- **Memory Usage**: Track memory consumption to prevent out-of-memory (OOM) crashes. When usage approaches 95%, you're in dangerous territory.

The key insight is that real-time monitoring prevents small issues from becoming big problems. A slight uptick in response time today could indicate a configuration issue that, if left unchecked, could cause a complete service outage tomorrow.

---

## Implementing Monitoring in Code

Translating monitoring concepts into actual code is straightforward. Here's how you can calculate essential metrics from your production logs:

```python
avg_latency = df["response_time_ms"].mean()
p95_latency = float(np.percentile(df["response_time_ms"], 95))
avg_tokens = df["tokens_generated"].mean()
error_rate = (df["_error_norm"] != "none").mean() * 100.0
avg_satisfaction = df["user_satisfaction"].mean()
```

This code snippet demonstrates calculating average latency, 95th percentile latency, average token usage, error rate, and user satisfaction from a dataframe containing production logs.

### Visualization for Quick Understanding

Creating graphs is essential for understanding at a glance what's happening with your model. Two particularly valuable visualizations are:

**Latency Distribution**: A histogram showing how response times are distributed across requests. This helps you identify whether you have a long tail of slow responses (which would affect your P99) or if most requests fall within acceptable ranges. Looking at a latency distribution, you might see most requests completing around 1000-1500ms, with a concerning spike at 3500ms that warrants investigation.

**Token Usage Distribution**: Understanding how users actually interact with your model informs your training data strategy. If you observe that most users generate between 200-600 tokens, but there's also a cluster of users generating 2000-2500 tokens, you'll want your post-training dataset to include examples representing both low and very high token distributions to match real usage patterns.

---

## The Model Production Cycle

Understanding where monitoring fits within the broader model lifecycle helps you design more effective observability systems. The production cycle consists of four stages:

### Experimentation Loop

This is where learning occurs. Your team produces candidate models through various training and fine-tuning approaches. At this stage, you're iterating quickly and tracking experimental metrics.

### Evaluation - Test

Candidate models are run against test sets and test environments. Models must pass predefined promotion rules before advancing. This might include accuracy thresholds, latency requirements, or safety evaluations.

### Staging

Models that pass evaluation are compared against the current production model on a small percentage of live traffic. This stage requires close monitoring to catch issues that didn't appear in synthetic test sets.

### Production

The fully deployed model serves all users. This stage focuses on behavioral observability, alerts, and maintaining a user feedback-to-data loop that feeds back into experimentation.

Production metrics monitoring spans both the staging and production phases, though the most critical observability happens in production where the model faces the full diversity of real-world inputs.

---

## Version Control and Rollback Strategy

When monitoring reveals something wrong in production, you need the ability to quickly roll back to a previous working version. This capability depends on rigorous version control—just like in regular software development.

### What to Version

For reliable rollbacks, you must freeze and track multiple components:

- **Dataset**: The exact training data used (e.g., `user_feedback_v1.2.jsonl`)
- **Configuration**: Hyperparameters and training settings (e.g., `hyperparams_v1.2.yaml`)
- **Code**: The exact codebase via git commit hash (e.g., `#a4b7c1da34brh4`)
- **Model Weights**: The trained checkpoint (e.g., `checkpoint_v1.2.pt`)
- **Results**: Evaluation metrics for reference (e.g., `accuracy_92.5%.json`)

### Rollback in Practice

Consider a scenario where Experiment v1.2 achieved 92.5% accuracy and is running stably in production. You deploy Experiment v1.3, which unexpectedly drops to 58.5% accuracy. With proper versioning, you can immediately roll back to v1.2's exact configuration—including the correct dataset version, hyperparameters, code, and model checkpoint.

### Recommended Tools

Several tools support this versioning workflow:

- **Git**: For code version control
- **DVC (Data Version Control)**: For tracking datasets and large files
- **HuggingFace Hub**: For model checkpoint management
- **MLflow**: For experiment tracking and model registry

The fundamental principle is simple: always have a backup that you can reliably restore.

---

## Systematic Monitoring Challenges

Beyond basic metrics, production models face several insidious challenges that require systematic monitoring to detect. These issues often develop gradually and may not be obvious until they've significantly degraded user experience.

### Data Drift

Data drift occurs when your input data changes over time because the world itself changes. Users start using different terms, new concepts become relevant, or external events shift behavior patterns.

A concrete example: after the COVID-19 pandemic, queries for "remote work" surged dramatically. A model trained on pre-pandemic data might not handle these queries well because the distribution of topics it encounters has shifted away from its training distribution.

Monitoring for data drift involves comparing the statistical properties of incoming requests against your training data distribution. When the distributions diverge significantly, it's time to collect new training data and retrain.

### Reward Collapse

Reward collapse is a particularly dangerous failure mode for models trained with reinforcement learning. It occurs when your model optimizes for a proxy metric at the expense of the actual goal you care about.

The classic example: if you train a content recommendation model to maximize click-through rate (CTR), it might learn to generate clickbait. The CTR goes up—the model is succeeding at its optimization target—but user satisfaction plummets because the content doesn't deliver on its promises.

The graph of reward collapse is unmistakable: CTR climbing steadily upward while user satisfaction (measured on a 1-10 scale) drops proportionally. You need to monitor both the proxy metric you're optimizing and the true outcome you care about.

### Data Degradation

Over time, multiple factors can cause gradual performance degradation:

- **Data Quality Issues**: Training data may become stale or corrupted
- **Infrastructure Problems**: Recent research highlights how subtle infrastructure issues can cause models to perform worse than they're actually capable of
- **Model Forgetting**: Through continued training or updates, models can lose capabilities they previously had

Data degradation is often hard to detect before you reach production scale. Small-scale tests might not reveal issues that only manifest with diverse, high-volume real-world traffic. This is why production monitoring is essential—some problems only become visible at scale.

---

## Multi-Model Evaluation Strategies

When you have multiple candidate models, you need systematic ways to compare them. Three complementary strategies help you understand which models perform best for different user segments and use cases.

### A/B Testing

A/B testing compares two model versions by segmenting live users. A portion of your traffic goes to Model A, while another portion goes to Model B, and you measure which performs better on your key metrics.

The power of A/B testing is that it evaluates business impact with real data, guiding deployment decisions. For example, you might test whether a chatbot with a more empathetic tone improves user satisfaction by routing a small percentage of users to the new model and measuring satisfaction improvements.

A/B testing happens in the production phase of your lifecycle, where you can compare passing models on actual user traffic.

### Side-by-Side Comparison

You may have seen this approach in products like ChatGPT, where users are shown two responses and asked which they prefer. Human evaluators compare output quality side-by-side, given identical inputs.

This approach provides quick, controlled quality checks without requiring large-scale user studies or extended time periods. Evaluators might judge two email responses and select the more professional-sounding one, providing immediate signal about model quality.

An interesting variation appears in email apps that suggest multiple possible responses—this naturally collects side-by-side preference data from users making genuine choices.

Side-by-side comparisons can happen in staging (with internal evaluators) or can be harvested from production to feed back into your experimentation loop.

### Playgrounds

Playgrounds are controlled environments where teams can explore model behavior, test limits, and discover hidden vulnerabilities. Internal teams input adversarial prompts, probe for edge cases, and identify failures or security risks like PII leaks.

The staging phase typically hosts internal playgrounds, while production might include beta-testing external playgrounds—limited early access for select users who can provide feedback on live canary traffic.

Playgrounds are invaluable for identifying issues that automated tests miss. Human creativity in finding edge cases often reveals failure modes that would otherwise only be discovered when they affect real users.

---

## Summary and Key Takeaways

Production monitoring for LLMs requires attention across multiple dimensions:

1. **Monitor the fundamentals**: Track performance (latency, tokens), cost (spending, GPU utilization), and reliability (format compliance, checkpoint loading, memory usage).

2. **Implement monitoring in code**: Calculate metrics from production logs and visualize distributions to understand user behavior patterns.

3. **Understand the production cycle**: Know where monitoring fits within experimentation, evaluation, staging, and production phases.

4. **Version everything**: Maintain the ability to roll back to any previous stable state by versioning datasets, configs, code, and model checkpoints together.

5. **Watch for systemic issues**: Data drift, reward collapse, and data degradation can silently erode model performance over time.

6. **Use multiple evaluation strategies**: A/B testing, side-by-side comparisons, and playgrounds each provide different insights into model quality.

With these systems in place, you'll be able to catch problems early, respond quickly when issues arise, and continuously improve your model based on real-world feedback. Now that you understand the model production lifecycle and monitoring essentials, you're ready to explore the infrastructure that makes all of this possible.

---

# M5-L06: Infrastructure for LLM Post-Training: A Production Guide

When moving from experimentation to production with large language models, infrastructure decisions become critical. This tutorial covers the essential considerations for deploying post-trained LLMs, including tool selection, model sizing, optimization techniques, routing strategies, and compute planning.

## Introduction to Infrastructure Considerations

Building production-ready LLM systems requires careful planning across four key dimensions. First, you need to select the right open-source tools for fine-tuning and reinforcement learning. Second, you must make decisions about model size, which cascades into virtually every other infrastructure choice. Third, if you're working with multiple models or adapters, you need a strategy for routing requests. Finally, your compute requirements will vary dramatically depending on your training approach and the scale of changes you're making to the model.

Each of these areas involves tradeoffs between flexibility, performance, cost, and ease of use. Understanding these tradeoffs helps you make informed decisions that align with your specific use case.

## Open-Source Fine-Tuning Tools

The open-source ecosystem offers several mature options for fine-tuning LLMs, each with distinct strengths suited to different stages of development.

### Hugging Face PEFT

Hugging Face's Parameter-Efficient Fine-Tuning (PEFT) library is often where practitioners begin their fine-tuning journey. It provides a flexible library that supports many different model architectures. The tradeoff is that it requires more coding and custom scripts to set up your training pipeline. This makes it particularly well-suited for research and experimentation, where you need the flexibility to try different approaches and modify the training process.

### LLaMA-Factory

For production fine-tuning, LLaMA-Factory has emerged as a strong choice. It offers a more turnkey framework with easy YAML configurations and built-in training pipelines. While it's focused primarily on LLaMA-family models, its integration with various inference serving frameworks has made it increasingly popular. If you need to quickly get a production-ready fine-tuned model, LLaMA-Factory reduces the engineering overhead considerably.

### Unsloth

Unsloth has carved out a niche by focusing aggressively on optimization. It delivers 2-5x faster training speeds while using approximately 30% less memory than alternatives. This optimization for speed and single-GPU usage makes it excellent for production-ready fine-tuning, especially when you're working with limited hardware or need rapid iteration cycles.

## Reinforcement Learning Tools

Reinforcement learning for LLMs is computationally intensive, often requiring distributed training across multiple GPUs. The tool landscape reflects this reality.

### Hugging Face TRL

Hugging Face's TRL (Transformer Reinforcement Learning) library provides a comprehensive RL library supporting many algorithms. Its seamless integration with the Transformers ecosystem makes it a natural choice if you're already using Hugging Face for other parts of your pipeline. Like PEFT, it's flexible enough for research and experimentation.

### Unsloth for RL

Unsloth extends its optimization advantages to reinforcement learning, offering 2-5x speed improvements and up to 80% less VRAM usage. It achieves this partly through efficient 4-bit RL training. If memory constraints are limiting your RL experiments, Unsloth can be a game-changer for production-ready deployment.

### ByteDance Verl

For enterprise-scale distributed training, Verl from ByteDance addresses the reality that RL often requires substantial compute resources. Given the memory demands of RL—which we'll discuss later—you frequently need multiple GPUs working in parallel. Verl supports open-source inference backends and provides advanced multi-GPU parallelism, making it suitable for large-scale RL training.

## Inference Serving Frameworks

After post-training your model, you need to serve it efficiently. Two frameworks currently dominate this space: vLLM and SGLang.

### vLLM

vLLM has become popular for its performance characteristics. It implements prefix caching, tensor parallelism, and pipeline parallelism to achieve high throughput with low latency. These optimizations are particularly valuable when serving models at scale.

### SGLang

SGLang offers complementary strengths, including structured generation capabilities and function calling for complex workflows. This makes it well-suited for agentic applications where the model needs to interact with external tools.

The two frameworks are essentially neck-and-neck in terms of features, and they continue to adopt each other's innovations. The recommendation is to try both and see which integrates more smoothly with your existing infrastructure and use case.

## Model Size: Choosing Between Big and Small

One of the most consequential decisions is selecting the size of your base model, as this determines downstream compute requirements and costs.

### The Case for Starting Small

The guidance is consistent: start with smaller models and measure performance across your metrics before scaling up. Smaller models (like an 8B parameter model) offer significant advantages in speed and cost, making them ideal for high-volume applications where you need to serve many requests affordably.

### Performance vs. Cost Tradeoffs

Comparing Llama 3.1 8B and 70B illustrates these tradeoffs clearly. The 70B model outperforms the 8B across benchmarks—scoring 86.0 vs 73.0 on MMLU (general knowledge), 84.5 vs 72.6 on HumanEval (code), 95.1 vs 80.5 on GSM8K (math), and 94.8 vs 83.4 on ARC Challenge (reasoning).

However, this performance comes at a cost. Input tokens cost approximately $0.23 per million for the 70B model versus $0.03 for the 8B—nearly an 8x difference. Output tokens show a similar pattern at $0.40 vs $0.05 per million. Speed differs dramatically too: the 8B model generates 173 tokens per second compared to 59 for the 70B model.

The takeaway: use 8B models for high-volume, cost-sensitive applications, and reserve 70B models for complex reasoning tasks where the additional capability justifies the expense.

## Optimization Techniques

When you need a larger model's capabilities but face hardware constraints, two techniques can help fit bigger models into smaller spaces.

### Quantization

Quantization reduces the numerical precision used to represent model weights. If your model uses 32-bit floating point (FP32) precision, dropping to 8-bit (INT8) or even 4-bit representation can reduce memory requirements by 2x, 4x, or even 8x. This massive reduction means you might fit the same model on cheaper hardware or accommodate larger models on existing GPUs.

The tradeoff is typically a small decrease in accuracy on your evaluation metrics. However, the impact varies—sometimes quantization causes minimal degradation, while other times it can reduce performance considerably. Always validate empirically with your specific use case before committing to a quantized model in production.

### Knowledge Distillation

Knowledge distillation takes a different approach: instead of compressing the same model, you train a smaller "student" model to mimic a larger "teacher" model. The teacher model generates predictions, including the full probability distributions over outputs, which provide rich supervision for training the student.

This probability information is particularly valuable because it captures not just what the teacher predicts, but how confident it is and what alternatives it considered. The student learns from all of this nuanced information, potentially achieving capabilities closer to the teacher than if it were trained from scratch on the same data.

As with quantization, you should expect some performance degradation and validate that the distilled model meets your requirements empirically.

## Multi-Model Routing

If you've developed multiple models or LoRA adapters for different tasks, a router can direct incoming queries to the appropriate model.

### LLM Model Router

A model router sits between user queries and your model pool. When a request arrives, the router analyzes it and directs it to the most suitable model—perhaps a specialized model for coding questions, another for general knowledge, and a third for reasoning tasks. The router can also aggregate responses from multiple models when that's beneficial.

### Multi-LoRA Serving

LoRA (Low-Rank Adaptation) creates an efficient opportunity for routing. Since LoRA adapters are small modifications to a base model, you can maintain a single set of pre-trained base weights while swapping different LoRA adapters based on the task. This architecture is more memory-efficient than maintaining multiple complete models and allows rapid switching between specialized capabilities.

## Compute Planning

Your compute requirements depend heavily on your post-training approach, with reinforcement learning and LoRA fine-tuning representing opposite ends of the spectrum.

### Reinforcement Learning: Heavy Compute

RL is computationally intensive in every dimension. Training takes longer, costs more, and requires substantial GPU capacity. The memory requirements often necessitate distributed training across multiple GPUs just to fit everything needed for the RL process. If you're doing RL, plan for significant infrastructure investment and longer iteration cycles.

### LoRA Fine-Tuning: Light Compute

LoRA fine-tuning offers a dramatic contrast. It's quick, low-cost, and has a targeted scope. You might even perform LoRA fine-tuning locally on an AI PC rather than renting cloud GPUs. This accessibility makes LoRA attractive for rapid experimentation and use cases where you need to create many specialized adapters.

### Scale of Changes Matters

Beyond the training method, consider how much you're changing in the model. More extensive modifications—more data, more parameters being updated, more training steps—require proportionally more compute to handle the forward passes, gradient calculations, and backpropagation through the model's parameters.

## Conclusion

Infrastructure decisions for LLM post-training involve interconnected choices that cascade through your entire system. Start by selecting tools that match your development stage—flexible options like Hugging Face for experimentation, more turnkey solutions like LLaMA-Factory or Unsloth for production. Choose model sizes that balance your performance requirements against cost and latency constraints, and consider quantization or distillation when you need to optimize further.

If you're working with multiple specialized models or adapters, implement routing to direct queries appropriately. Finally, plan your compute based on whether you're doing heavy RL training or lighter LoRA fine-tuning, and account for the scale of changes you're making to the model.

The key principle throughout is empirical validation: theoretical performance gains from optimization techniques must be verified against your specific metrics and use case before deployment.

---

# M5-L07: GPU Infrastructure for LLM Training and Inference: A Practical Guide

## Introduction

Deploying large language models requires careful consideration of hardware infrastructure—both for training and serving models in production. This guide covers the essential factors in choosing GPUs, understanding memory requirements across different training approaches, and planning capacity for inference workloads. By the end, you'll understand how to estimate GPU needs for fine-tuning, LoRA, reinforcement learning, and production inference, along with the associated costs.

---

## GPU Selection Fundamentals

When choosing a GPU for machine learning workloads, memory is often the primary constraint. During training, four major components must fit in GPU memory simultaneously: the model weights, the optimizer states, the activations (intermediate values computed during the forward pass), and the gradients (computed during backpropagation). All of these must coexist on the GPU for training to proceed.

When a model is too large to fit on a single GPU, you need to distribute it across multiple devices—a technique called **model sharding**. While this enables training of larger models, it introduces complexity and communication overhead between GPUs, which can impact training speed.

Beyond memory, cost and time represent the other key tradeoffs. Training duration directly affects cost, but there's also the question of how many experiments you can run in parallel. During error analysis and hyperparameter tuning, running more parallel experiments helps you converge on the right training recipe faster. Using additional compute to scale out parallel experiments reduces time but increases upfront costs. The fundamental tradeoff is often between higher upfront costs to parallelize across many GPUs versus longer run times on cheaper hardware.

---

## Precision and Memory Requirements

The amount of memory a model consumes depends directly on the numerical precision used to represent its parameters. The relationship is straightforward:

**Memory = Number of Parameters × Bytes per Parameter**

For a 7 billion parameter model, here's how memory requirements scale with precision:

| Precision | Bytes per Parameter | Memory for 7B Model |
|-----------|---------------------|---------------------|
| FP32 (float32) | 4 bytes | 28 GB |
| FP16 / BF16 | 2 bytes | 14 GB |
| INT8 | 1 byte | 7 GB |
| INT4 | 0.5 bytes | 3.5 GB |

This dramatic reduction—from 28 GB down to just 3.5 GB—illustrates why quantization is so valuable. However, an important caveat: quantization is typically recommended **after** training is complete, specifically for inference. During training, you want full precision to accurately represent gradients and weight updates. Quantization works best when you've validated that the lower-precision model maintains acceptable performance compared to the original.

### Inference vs. Training Memory

The memory requirements differ substantially between inference and training:

**For inference**, you primarily need to store the model parameters (approximately 1× the parameter count in your chosen precision) plus the **KV cache**. The KV cache is an optimization technique that stores the key and value matrices from the attention mechanism, allowing more efficient inference by avoiding redundant computation. The cache size depends on prompt length and batch size.

**For training with AdamW**, memory requirements balloon to roughly **4× the parameters** plus activations. This breaks down as:
- 1× for the model weights
- 1× for the gradients
- 2× for the two optimizer states (momentum and variance in AdamW)
- Plus additional memory for activations

This applies to both fine-tuning and reinforcement learning, though RL has additional considerations we'll explore later.

---

## Memory-Saving Techniques: LoRA and QLoRA

When working with limited GPU memory (VRAM), parameter-efficient fine-tuning methods offer substantial savings.

**LoRA (Low-Rank Adaptation)** works by freezing the base model and only training small adapter layers. Since the base model weights don't change, you don't need to store gradients or optimizer states for them—only for the adapter weights. This can reduce memory requirements to roughly **10-20% of full fine-tuning**, sometimes even less. You still need to store the frozen base model in memory, but eliminating the gradient and optimizer overhead for most parameters yields significant savings.

**QLoRA** takes this further by also quantizing the frozen base model (typically to 4-bit precision). Since you're not updating those weights anyway, representing them in lower precision has no impact on training dynamics while dramatically reducing the base model's memory footprint.

These techniques are particularly valuable because they enable training on mid-tier GPUs that couldn't otherwise handle full fine-tuning of large models. GPUs with larger memory capacity can fit bigger models without sharding, simplifying the training setup and avoiding inter-GPU communication overhead.

---

## Memory Requirements by Training Method

Different training approaches have vastly different memory profiles. Understanding these differences is crucial for planning your infrastructure.

### Fine-Tuning

Full fine-tuning requires the complete memory stack: parameters, gradients, optimizer states, and activations. For models with 13 billion parameters or more, you often need well over 80 GB of VRAM per GPU—typically requiring high-end GPUs like the NVIDIA A100 (80GB) or H100, or distributing across multiple devices.

### LoRA

LoRA dramatically reduces requirements by only storing adapter weights and their associated gradients/optimizer states, while keeping the base model frozen. The memory footprint is typically 10-20% of full fine-tuning, making it accessible on mid-tier GPUs.

### Reinforcement Learning

RL training is the most memory-intensive approach because you need **multiple models in memory simultaneously**. For PPO (Proximal Policy Optimization), this typically includes:
- The policy model being trained (with full gradients and optimizer states)
- A reference model (frozen, for computing KL divergence)
- A value model or critic
- A reward model

**GRPO (Group Relative Policy Optimization)** offers some relief by eliminating the need for a separate baseline estimation model, but it's still substantially more demanding than supervised fine-tuning.

### GRPO Memory Breakdown

Let's examine a concrete example: training a 13B parameter model with GRPO using FP16 precision.

| Component | Memory Requirement |
|-----------|-------------------|
| Policy LLM (full training) | ~104 GB (26 GB × 4 for weights + grads + 2 optimizer states) |
| Reference LLM (frozen) | ~26 GB (inference only, no gradients) |
| Reward Model (~7B) | ~14 GB |
| Activations (from group rollouts) | ~30-50 GB (depends on batch × sequence length) |
| **Total** | **~170-190 GB VRAM** |

This is a tremendous amount of memory for what's considered a "small" 13B model. The activations component is particularly variable because GRPO performs group rollouts—generating multiple completions per prompt to compute advantages—which scales with batch size and sequence length.

### Summary Comparison

| Aspect | Fine-Tuning | LoRA | RL |
|--------|-------------|------|-----|
| **Memory** | Full stack required | Adapters only; fits mid-tier GPUs | Multiple copies; biggest memory hog |
| **Compute** | High throughput training | Moderate; efficient training | Extreme compute + throughput required |

---

## Inference Capacity Planning

Once your model is trained, you need infrastructure to serve it in production. Capacity planning for inference involves three key considerations: traffic, throughput, and hardware allocation.

### Traffic Estimation

User traffic is calculated as:

**Traffic = QPS × Average Request Tokens**

Where request tokens include input tokens, context (like system prompts or retrieved documents), any tool outputs, and the tokens you'll generate as output. Accurately estimating this requires understanding your application's usage patterns.

### Throughput Planning

Throughput planning involves determining your batch size and expected tokens per second per GPU. These values depend on your model size, precision, and the specific GPU hardware. Batching multiple requests together improves GPU utilization but increases latency for individual requests.

### Hardware Allocation

For hardware sizing, you need to count GPUs to meet your p95 (95th percentile) latency target while including headroom for canary deployments and traffic spikes.

### Example Sizing Calculation

Consider this production scenario:
- **Target**: 30 queries per second (QPS)
- **Average prompt size**: 1,200 tokens
- **Average generated output**: 300 tokens
- **Latency requirement**: p95 under 900 milliseconds

With batching and FP8 precision, assume approximately 60 tokens per second per GPU at the p95 latency target.

**GPU requirement**: ~10 GPUs, plus 30% headroom for canary deployments

**Cost estimation** (at $1.99/hour per GPU):
- $19.90 per hour
- $477.60 per day
- **$14,328 per month**

The cost formula for estimating price per 1,000 tokens is:

```katex
\text{cost per 1k tokens} \approx \frac{\text{\$/GPU-hr}}{\text{tokens/s/GPU} \times 3600} \times 1000
```

### Practical Considerations

Accurate upfront prediction is challenging—user behavior is often unpredictable, and traffic patterns may vary significantly by time of day or day of week. This makes **elastic sizing** essential: use cloud compute options that allow you to scale up during peak demand and scale down during quiet periods. This flexibility helps optimize costs while maintaining service quality.

---

## Key Takeaways

1. **Memory is usually the primary constraint** for GPU selection. The model, optimizer states, activations, and gradients must all fit in VRAM during training.

2. **Precision significantly impacts memory**: A 7B model ranges from 28 GB (FP32) to 3.5 GB (INT4). Use quantization for inference after training, not during.

3. **Training methods have vastly different memory profiles**: Full fine-tuning requires the complete stack, LoRA reduces this to 10-20%, and RL can require 2-4× fine-tuning memory.

4. **RL training is exceptionally demanding**: GRPO on a 13B model needs 170-190 GB of VRAM due to multiple models and activation storage from group rollouts.

5. **Inference capacity planning requires estimating traffic, throughput, and latency targets**. Build in headroom for canary deployments and consider elastic scaling for cost optimization.

6. **Cost adds up quickly**: Even a modest 30 QPS deployment can cost over $14,000/month in GPU infrastructure. Plan accordingly and optimize where possible.

Understanding these infrastructure requirements is essential before deploying models to production. Proper planning prevents costly surprises and ensures your system can meet user demand reliably.

---

# M5-L08: Production-Ready Checklist for Machine Learning Models

## Introduction

Deploying a machine learning model to production is more than just training a high-performing model—it requires careful preparation across multiple dimensions to ensure reliability, reproducibility, and continuous improvement. This tutorial walks through a comprehensive production checklist that every ML practitioner should complete before shipping their model into the real world.

By the end of this guide, you'll understand the five critical areas to address before production deployment: reproducible configuration, promotion and rollback policies, monitoring and observability, feedback pipelines, and infrastructure allocation.

---

## 1. Clear Reproducible Model Configuration

The first item on your production checklist is ensuring you have a clear, reproducible model configuration. This is foundational because you need to trust the model you're deploying.

### Why Reproducibility Matters

When you release a model into production, you must be able to reproduce it exactly. This means anyone on your team should be able to re-run your evaluation suite on held-out test sets and get the same numbers—within some reasonable confidence interval tolerance. If you can't reproduce your results, you can't trust that the model in production is actually the model you evaluated.

Reproducibility also enables rollback. If something goes wrong in production, you need to be able to return to the exact artifact that was previously working. Without proper versioning and configuration management, rollbacks become guesswork.

### What to Include in Your Configuration

A robust model configuration should capture every component that affects model behavior:

```json
{
  "llm_sha": "LLM@abc123",
  "adapters": ["lora_med_safety_v2@e91f", "lora_math_v1@bb07"],
  "input_sha": "INPUT@p1e3",
  "tokenizer_sha": "TOK@t99",
  "sampling": {"temperature": 0.2, "top_p": 0.9},
  "reward_model_sha": "RM_EVAL@9af2",
  "verifiers": {"json_schema": "v4.3", "span": "v2.1", "unit_tests": "v2.0"},
  "tools": {"search_api": "3.2.1", "fixtures": "search_v3.json", "repo_sha": "GIT_SHA_CODEBASE_V3"}
}
```

This configuration captures the base model version, any adapter weights (like LoRA fine-tunes), input processing, tokenization, sampling parameters, evaluation models, verification tools, and external dependencies. Every SHA or version string represents a specific, immutable artifact that can be retrieved and redeployed.

---

## 2. Promotion Rules and Rollback Conditions

The second checklist item involves establishing clear promotion rules and rollback conditions across your deployment stages: development, staging, and production.

### Turning Your North Star into a Contract

Throughout model development, you've been guided by a "North Star"—the key metrics and behaviors that define success for your use case. At the production gate, this North Star must be transformed from an aspirational goal into a coded go/no-go contract.

This means defining explicit thresholds: What accuracy must the model achieve? What safety checks must it pass? What latency requirements must it meet? By codifying these requirements, you remove ambiguity from deployment decisions and enable automation.

### The Three Components of Promotion Policy

A complete promotion and rollback system includes three elements:

**Coded go/no-go contracts**: Your North Star metrics become automated gates. If a model meets all criteria, it can proceed to the next stage. If it fails any criterion, promotion is blocked. This removes subjective judgment from critical deployment decisions.

**Light guardrails in staging**: Before reaching production, models should run in a staging environment with guardrails—safety checks, output filters, or constraint systems that catch problematic behaviors. Staging lets you validate model behavior on realistic traffic without risking production users.

**Pre-agreed rollback triggers**: Before deploying, decide what conditions will trigger an automatic rollback. These might include error rate spikes, latency degradation, safety violations, or business metric declines. Pre-agreeing on these triggers ensures fast response to production issues without requiring real-time debate about whether to roll back.

### Benefits of Explicit Promotion Rules

When you codify your promotion rules and write them down, you gain the ability to automate much of the deployment pipeline. Models can flow from development through staging to production with minimal human intervention, while still maintaining rigorous quality gates. When issues arise, rollback can happen automatically and instantly.

---

## 3. Monitoring and Observability

Once your model is in production, you need visibility into how it's performing. The third checklist item ensures you have monitoring and observability systems in place.

### Behavioral and System Metrics

Effective monitoring covers two categories of metrics:

**Behavioral metrics** track what the model is doing—response quality, accuracy on specific tasks, safety violations, user satisfaction signals, and output characteristics. These tell you whether the model is serving users well.

**System metrics** track how the model is running—latency, throughput, error rates, GPU utilization, and resource consumption. These tell you whether the model is operationally healthy.

### Slice-Based Analysis

Not all users or queries are equal. Your monitoring should break down performance by "slices"—meaningful segments of your traffic that matter for your use case. For example, you might monitor separately for different user types, query categories, geographic regions, or difficulty levels.

Each slice should have its own Service Level Objectives (SLOs) covering latency, cost, and throughput. A slice that meets all SLOs gets a "go" status; one that fails any objective gets a "no go" status and requires attention.

### Making Behavior Visible and Actionable

The goal of monitoring is not just visibility—it's actionability. Your dashboards and alerts should make it obvious what's working, what's failing, and what actions to take. When a slice shows degraded performance, your monitoring system should help you understand why and decide how to respond.

---

## 4. Feedback to Data Pipeline

The fourth checklist item addresses the ongoing improvement cycle: establishing a feedback-to-data pipeline that captures production insights for future model generations.

### The Data Flywheel

Production deployment isn't the end of model development—it's the beginning of a new data collection opportunity. Users interacting with your model generate valuable signals: which responses worked, which failed, which edge cases appeared, and which behaviors need adjustment. A well-designed feedback pipeline transforms these signals into training data for your next model iteration.

### The Four-Stage Pipeline

The feedback-to-data pipeline follows four stages:

**Mine**: Examine your production logs to identify interesting cases. Look for failures, unexpected behaviors, user complaints, and also successes worth reinforcing. Cluster similar cases together to understand patterns.

**Triage**: Not all mined examples are equally valuable. Triage involves prioritizing which cases to address—focusing on high-impact failures, common patterns, or strategically important behaviors.

**Synthesize for coverage**: Often, a single production failure represents a broader capability gap. Use synthetic data generation (the "k→1" expansion mentioned in the slides) to create many training examples that cover the underlying issue. This amplifies the signal from sparse production feedback.

**Queue for training**: Feed the processed data into your training pipeline, whether for supervised fine-tuning or reinforcement learning. The cycle then repeats: improved models go to production, generate new feedback, and drive further improvements.

### High-Signal Supervision

The key insight is that production failures provide extremely high-signal supervision for model improvement. Unlike generic training data, these examples come directly from real use cases where your model fell short. By systematically capturing and processing this feedback, you create a continuous improvement loop that makes each model generation better than the last.

---

## 5. Infrastructure Allocation

The final checklist item ensures you have appropriate infrastructure to serve your model at production scale.

### Balancing Three Constraints

Infrastructure allocation requires balancing three key factors:

**Throughput**: How many requests can your system handle? Your infrastructure must support expected traffic volumes, including peak loads and potential growth.

**Latency**: How fast does each request complete? Many applications have strict latency requirements—if responses are too slow, user experience suffers or downstream systems time out.

**Cost**: What does it cost to run your model? GPUs are expensive, and inefficient infrastructure allocation can make otherwise valuable models economically unviable.

### Elastic Scaling

Production traffic is rarely constant. You need infrastructure that can scale elastically—expanding to handle traffic spikes and contracting during quiet periods to save costs. This requires planning for GPU provisioning, load balancing, and autoscaling policies.

### Preserving Model Behavior

A critical but sometimes overlooked consideration: infrastructure changes can affect model behavior. Quantization for faster inference, batching strategies for higher throughput, or different hardware configurations might subtly change model outputs. As you optimize infrastructure, verify that you're not regressing on important behaviors.

The goal is to meet your targets on GPU scale-out elastically, without regressing on the important behaviors your evaluation suite measures.

---

## Conclusion: Your Complete Production Checklist

Before deploying your model to production, verify that you have addressed all five areas:

1. **Clear reproducible model config**: Anyone can re-run evaluations and get the same numbers; anyone can roll back to exact artifacts within CI tolerance.

2. **Promotion rules and rollback conditions**: Your North Star is converted to coded go/no-go contracts, staging guardrails are in place, and automatic rollback triggers are pre-agreed.

3. **Monitoring and observability**: Both behavioral and system metrics are tracked across meaningful slices, with SLOs defined for latency, cost, and throughput.

4. **Feedback-to-data pipeline**: Production failures and successes are mined, triaged, synthesized for coverage, and queued for training—creating a continuous improvement flywheel.

5. **Infrastructure allocation**: Resources are provisioned to meet ROI and scaling requirements, with elastic scale-out that doesn't compromise model behavior.

This checklist represents the bridge between successful model training and successful model deployment. By systematically addressing each item, you ensure that your model is not just good in evaluation, but reliable, maintainable, and improvable in production.

With your checklist complete, you're ready to roll out your model. Good luck with your deployment!

---