# Banking correction example with Agent Skills

This example shows how Zep creates an Agent Skill from a successful teacher Trace.
A PydanticAI student agent then reads the live Skill and repeats a banking procedure.
The tools, input models, database, and reports use Pydantic.

The bank policies and account records are invented. This example is inspired by a
banking interest-correction task, but it does not include tau3 code or data.
Its scores are not tau3 benchmark results. A new teacher run or Skill compilation
can produce a different result. The example does not guarantee a fail-to-pass change.

## Requirements

- Use Python 3.12 or 3.13. The setup command below lets uv install Python 3.12.
- Install [Git](https://git-scm.com/downloads) and
  [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Use an Anthropic API key with access to `claude-haiku-4-5-20251001` and
  `claude-opus-5-5`, or select models that your account can use.
- Use a Zep project API key with Agent Skills access on the
  [public Zep service](https://app.getzep.com/). An account without Agent Skills
  access receives HTTP 403. Use a separate project for this example.
- Use an internet connection to install dependencies and call Anthropic and Zep.

Model calls and Skill compilation use paid services. The main command runs three
model episodes and uploads the passing teacher Trace to your Zep project.
The command creates a new Agent, Trace, and Skill. These resources remain in Zep
after the command ends. The example does not delete them automatically.

The example needs no Docker, Node.js, customer simulator, or shell sandbox.
The checked platform is Linux x86_64. Native macOS and ARM64 execution are unverified.
The optional verification command uses Unix file locking and does not support native Windows.

## Install from a clone

1. Clone the repository:

   ```bash
   git clone https://github.com/getzep/zep.git
   ```

2. Enter the example directory:

   ```bash
   cd zep/examples/python/agent-skills-banking
   ```

3. Install the locked dependencies and local check tools:

   ```bash
   uv sync --frozen --python 3.12 --group dev
   ```

4. Check the local code before you provide API keys:

   ```bash
   uv run --frozen pytest -q
   ```

   The tests use local fixtures and mock services. They do not call a model or Zep.

## Set the API keys

Use Bash for these commands. They read each key without terminal echo or a literal
key in your shell history. Do not put API keys in source files or shared transcripts.

1. Read and export the Anthropic key:

   ```bash
   read -rsp 'Anthropic API key: ' ANTHROPIC_API_KEY; echo
   export ANTHROPIC_API_KEY
   ```

2. Read and export the Zep project key:

   ```bash
   read -rsp 'Zep API key: ' ZEP_API_KEY; echo
   export ZEP_API_KEY
   ```

For another shell, set the same environment variables through its secure input
method. The example reads environment variables; it does not load a `.env` file.
The Zep client uses the public US production service by default.
For an EU project, set `ZEP_BASE_URL` to `https://api.eu.getzep.com`.

## Run the learning loop

Run this command from the example directory:

```bash
uv run --frozen python walkthrough.py run --output runs/live-001
```

The command will:

1. Run Haiku without a Skill on a fresh local student case.
2. Run Opus at maximum effort on a separate teacher case.
3. Compare the teacher's complete database with the expected database.
4. Create a new Agent in your Zep project if the teacher passes.
5. Upload the teacher Trace with external verification and wait for Trace closure.
6. Wait for Zep to compile and admit a searchable Skill.
7. Print the Skill UUID, version, Markdown, and SHA-256 hash.
8. Run the same student configuration on its original case with the live Skill.
9. Print each score, token count, episode time, and failed provider-call count.

Allow several minutes for the model episodes and Skill compilation.
The default timeout for Trace closure and compilation polling is 900 seconds each.
The final output includes a summary for `without_skill`, `teacher`, and `with_skill`.
It also saves the summary under `runs/live-001/summary.json`.
The teacher must pass before a successful Trace is uploaded.
The student must read a retrieved Skill for its Skill-assisted run to be valid.
The generated Skill is not edited or replaced with a local procedure.

Haiku uses disabled thinking and temperature 0 in both student configurations.
Both configurations receive the same task, policies, tools, and limits.
The live Skill catalog is the only treatment difference.
Opus uses maximum effort and the provider-default temperature.
The teacher also writes a package-comparison table and a reusable procedure.
That teaching request is not part of the student prompt.

If your provider account does not support a default model, start a new run with
supported model names and compatible effort settings. For example:

```bash
uv run --frozen python walkthrough.py run --output runs/custom-001 \
  --student YOUR_STUDENT_MODEL --teacher YOUR_TEACHER_MODEL --effort none
```

Changing models can change the result. Use `walkthrough.py run --help` to see the options.

## Understand the task and score

The task has two savings accounts and several checking and card products.
The agent must find eligible, permitted packages, select the package with the
greatest interest-dollar benefit, post one correction credit per underpaid account,
and save one complete discrepancy report per credit.

The policies apply different checking and card balance caps, qualification rules,
and package exclusions. Both student configurations have a `quote_interest` tool.
That tool evaluates one proposed package and calculates the correction amount.
It does not select the best package or write a credit or report.

The evaluator compares the complete final database, including report fields.
A correct final message cannot replace correct database records.
The example tests package selection and completion of the required writes.
It does not test arithmetic accuracy. It does not promise lower token use or latency.

The synthetic policies are in `banking.py` under `DOCUMENTS`. The `make_case(seed)`
function generates account records locally. No dataset download or internal Zep
service is required. The default teacher seed is 54; the student seed is 404.
The run saves both cases and the policies in `cases.json` for inspection.

## Inspect the output

| Path under the run directory | Content |
| --- | --- |
| `summary.json` | Scores and resource use for the three episodes |
| `cases.json`, `config.json` | Synthetic input, model configuration, source hashes, and dependency versions |
| `agent.json`, `upload.json` | Agent UUID, teacher Trace UUID, and confirmed Trace closure |
| `skill-*-v*.md`, `skill-pin.json` | Retrieved Skill Markdown and its pinned identity |
| `learning-progress.jsonl`, `zep-calls.jsonl` | Compilation progress and public API status records |
| `without-skill/`, `teacher/`, `with-skill/` | Messages, tools, model settings, usage, and final database for each episode |
| Each episode's `evaluation.json` | Expected database, actual database, and exact differences |
| `failure.json`, `run-failure-*.json` | Error details when an episode or command stops |

The runner does not save API keys. It does save full synthetic conversations and
retrieved Skill content. Review run files before sharing them. The local
`.gitignore` excludes the default `runs/` directory and environment files.

## Recover from an interrupted run

A provider error, public API error, timeout, or incomplete episode is an invalid run,
not a model task failure. Requests have no automatic provider retries.

- If compilation or Trace closure reaches the wait limit, keep the run directory.
  Repeat the same command with `--resume`:

  ```bash
  uv run --frozen python walkthrough.py run --output runs/live-001 --resume
  ```

  The command reuses completed model stages. It resumes confirmed Trace closure
  through read-only polling. Use the same source, dependencies, and command options.
- If a model-stage directory is incomplete, the command refuses to overwrite it.
  Preserve the failed attempt and start with a new output directory.
- If Trace event upload was interrupted, inspect `upload-start.json` and the saved
  Trace in your Zep project before manual recovery. Do not upload duplicate events.
- For HTTP 401, check the key and service endpoint. For HTTP 403 from Zep, check
  the project's Agent Skills access. Also confirm that your model key permits
  the selected models before you start another paid run.

## Optional paired checks

These commands make additional paid model and Zep calls. They require a completed
learning loop. They retrieve the live Skill again and reject a changed catalog.

For three new pairs on the same student case:

```bash
uv run --frozen python walkthrough.py verify \
  --learned-run runs/live-001 --output runs/repeat-001 --repeat-case --count 3
```

Each episode makes new model calls with a fresh database. The order alternates
between pairs. An append-only ledger reserves run indices before calls start.
For another check, use a new output directory and `--first-run 3` after this example,
including if the first check was interrupted. Preserve every attempt.

For an exploratory check on three new generated cases:

```bash
uv run --frozen python walkthrough.py verify \
  --learned-run runs/live-001 --output runs/fresh-001 --first-seed 10000 --count 3
```

Use new, non-overlapping seeds and a new output directory for later fresh-case checks.
The command excludes teacher, student, screening, and previously reserved seeds.
The summary's strict `accepted` flag requires every baseline to fail and every
Skill-assisted episode to pass. The flag is specific to this demonstration goal.

Repeated runs of one selected case do not establish performance on unseen tasks.
Temperature 0 does not guarantee deterministic outputs or independent samples.
The conditional lower bound in the summary assumes independence; the example does
not establish that assumption. Fresh-case checks are exploratory and can include
baseline successes. One successful learning loop does not guarantee future results.

## Local code checks

These commands need no API keys or paid services:

```bash
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen pyright
uv run --frozen python walkthrough.py run --help
uv run --frozen python walkthrough.py verify --help
```
