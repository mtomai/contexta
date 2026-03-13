# Contributing to Contexta

First of all, thank you for your interest in contributing to **Contexta**! 🎉

Every contribution is valuable: whether it's reporting a bug, proposing a new feature, improving the documentation, or writing code. This guide will help you get started in the best way.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting a Bug](#reporting-a-bug)
  - [Proposing a New Feature](#proposing-a-new-feature)
- [Development Environment Setup](#development-environment-setup)
  - [Prerequisites](#prerequisites)
  - [Backend (Python)](#backend-python)
  - [Frontend (Node.js)](#frontend-nodejs)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Creating a Pull Request](#creating-a-pull-request)
- [Code Style Conventions](#code-style-conventions)
  - [Python (Backend)](#python-backend)
  - [JavaScript/React (Frontend)](#javascriptreact-frontend)
  - [Commit Messages](#commit-messages)

---

## Code of Conduct

This project adopts the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to abide by its terms. Report unacceptable behavior by opening an issue.

---

## How Can I Contribute?

### Reporting a Bug

If you've found a bug, open a [new Issue](../../issues/new) including:

1. **Clear and descriptive title** of the problem.
2. **Steps to reproduce** (step-by-step).
3. **Expected behavior** vs. **actual behavior**.
4. **Screenshots or logs** (if applicable).
5. **Environment**: Operating system, Python/Node.js version, browser used.

### Proposing a New Feature

Ideas are always welcome! Open a [new Issue](../../issues/new) with:

1. **Clear description** of the proposed feature.
2. **Motivation**: what problem it solves or what value it adds.
3. **Possible implementation** (optional, but useful for discussion).

> **Note:** Before starting work on a major feature, open an issue to discuss it with the maintainers. This avoids duplicate work or directions not aligned with the project.

---

## Development Environment Setup

### Prerequisites

- **Git**
- **Python 3.10+** (recommended: use [uv](https://docs.astral.sh/uv/) as package manager)
- **Node.js 18+** (with npm)
- A valid **OpenAI API key**

### Backend (Python)

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/contexta.git
   cd contexta
   ```

2. **Create the virtual environment and install dependencies:**

   With `uv` (recommended):
   ```bash
   cd backend
   uv sync --extra dev
   ```

   Or with `pip`:
   ```bash
   cd backend
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Download the required NLTK module (punkt):**

   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

   > This is required for the BM25 search engine with Italian stemming.

4. **Configure environment variables:**

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file by entering your OpenAI API key and other settings:

   ```dotenv
   OPENAI_API_KEY=your_api_key_here
   CHROMA_DB_PATH=./chroma_db
   UPLOADS_PATH=./uploads
   ALLOWED_ORIGINS=["http://localhost:5172","http://127.0.0.1:5172","http://localhost:3000"]
   ```

5. **Start the backend in development mode:**

   ```bash
   uv run uvicorn app.main:app --reload --port 8000
   ```

6. **Run the tests:**

   ```bash
   uv run pytest
   ```

### Frontend (Node.js)

1. **Install dependencies:**

   ```bash
   cd frontend
   npm install
   ```

2. **Configure environment variables:**

   ```bash
   cp .env.example .env
   ```

   Make sure the `.env` file contains the backend URL:

   ```dotenv
   VITE_API_URL=http://127.0.0.1:8000
   ```

3. **Start the frontend in development mode:**

   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5172`.

---

## Development Workflow

### Branching Strategy

1. Create a branch from `main` with a descriptive name:

   ```bash
   git checkout -b feat/feature-name    # for new features
   git checkout -b fix/bug-description  # for bugfixes
   git checkout -b docs/what-changed    # for documentation
   ```

2. Make frequent and atomic commits on your branch.
3. Keep your branch up to date with `main`:

   ```bash
   git fetch origin
   git rebase origin/main
   ```

### Creating a Pull Request

1. **Make sure everything works** before opening the PR:
   - The linter reports no errors.
   - Existing tests pass.
   - You have written tests for the new code (if applicable).
   - The frontend build succeeds (`npm run build`).

2. **Push your branch:**

   ```bash
   git push origin feat/feature-name
   ```

3. **Open a Pull Request** targeting the `main` branch on GitHub.

4. **In the PR description**, include:
   - Reference to the related issue (e.g., `Closes #42`).
   - Description of the changes made.
   - Screenshots (if there are UI changes).

5. **Wait for the review.** A maintainer will review your code and may request changes.

---

## Code Style Conventions

### Python (Backend)

- **Linter/Formatter:** Use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

  ```bash
  # Check for style errors
  ruff check backend/

  # Auto-format the code
  ruff format backend/
  ```

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type hints for public function signatures.
- Async functions must use the `async def` prefix.

### JavaScript/React (Frontend)

- **Linter:** Use [ESLint](https://eslint.org/) if configured in the project.
- **Formatter:** Use [Prettier](https://prettier.io/) for automatic formatting.
- Use functional React components with hooks.
- Prefer `const` over `let`, avoid `var`.

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) convention:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

**Common types:**

| Type       | Description                            |
|------------|----------------------------------------|
| `feat`     | New feature                            |
| `fix`      | Bug fix                                |
| `docs`     | Documentation changes                  |
| `style`    | Formatting (no logic changes)          |
| `refactor` | Code refactoring                       |
| `test`     | Adding or modifying tests              |
| `chore`    | Maintenance, dependencies, CI/CD       |

**Examples:**

```
feat(backend): add hybrid BM25+vector search
fix(frontend): resolve chat scroll issue on mobile
docs: update CONTRIBUTING with setup instructions
```

---

Thank you again for wanting to contribute to Contexta! If you have questions, don't hesitate to open an issue or contact the maintainers.
