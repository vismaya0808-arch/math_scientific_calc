# MathPilot — Scientific Learning Studio

**An Integrated Mathematics Platform for Calculation, Exploration, and Offline Learning**

## About the Project

MathPilot — Scientific Learning Studio is an interactive mathematics learning platform that combines a scientific calculator with an offline mathematics learning environment. It is designed to help students move beyond obtaining numerical answers and develop a better understanding of the mathematical concepts and reasoning behind them.

Students often switch between calculators, graphing tools, learning resources, and practice platforms while studying mathematics. MathPilot brings these learning activities together in a unified application, allowing students to calculate, solve, visualise, practise, and review their progress in one place.

The platform combines scientific computation, symbolic mathematics, graphical exploration, guided learning resources, practice questions, and local progress tracking. Its Offline Lab is designed to support core learning activities without requiring an internet connection after the necessary software and dependencies have been installed.

## Problem Statement

Mathematics learning is not only about finding the correct answer; it is also about understanding the process used to reach it. Students may struggle to understand formulas, follow mathematical steps, interpret graphs, or identify topics that require additional practice.

Using separate tools for calculations, visualisation, and revision can make the learning process fragmented. Dependence on online resources can also limit access in environments with unreliable internet connectivity.

MathPilot addresses these challenges by integrating mathematical tools and learning activities into a single platform that supports independent practice, conceptual exploration, and revision.

## Objectives

- Make mathematical calculations and exploration accessible through one application.
- Help students understand mathematical operations and symbolic solutions.
- Support visual learning through interactive mathematical graphs.
- Provide topic-wise learning resources, examples, and practice questions.
- Enable students to review their practice history and identify areas for improvement.
- Support offline access to core learning features after installation.

## Features

### 1. Scientific Calculator

The Scientific Calculator provides a central interface for performing supported mathematical and scientific calculations. It is designed for students who need a convenient tool for coursework, assignments, and everyday mathematical problem-solving.

### 2. Learn — Topic-Based Learning

The Learn section provides structured mathematical learning material across seven topics, including explanations, formulas, and worked examples. Students can use these resources to revise concepts and understand how mathematical rules are applied.

### 3. Solve — Symbolic Mathematics

The Solve section uses SymPy to perform supported symbolic mathematical operations, including:

- Simplification of expressions.
- Factorisation and expansion.
- Differentiation.
- Integration.
- Solving supported equations.

These capabilities allow students to explore mathematical expressions and examine symbolic results rather than relying exclusively on numerical calculations.

### 4. Explore — Interactive Graphs

The Explore section allows students to investigate supported mathematical functions through interactive plots. It also supports derivative visualisation, helping learners explore how functions change and how their mathematical properties relate to their graphs.

### 5. Practice — Questions and Feedback

The Practice section provides mathematical questions for independent revision. Students can attempt questions, use available hints, check their answers, and review explanations to strengthen their understanding.

### 6. My Progress — Learning History

MathPilot includes local progress tracking to help students review their learning activity. The progress section uses SQLite to store practice attempts and supports features such as:

- Reviewing previous attempts.
- Viewing topic-wise accuracy.
- Identifying topics that may need revision.
- Exporting practice records to CSV.

This allows students to monitor their practice over time without requiring a separate online account for the local progress database.

### 7. Offline Mathematics Learning

The Offline Lab is designed to make its core learning features available locally after the required packages and application files have been installed. Topic guides, supported symbolic calculations, graph exploration, practice activities, and locally stored progress can operate without continuous internet access.

Optional online AI or OCR capabilities, if configured, may require additional dependencies, services, or internet access.

### 8. Integrated Learning Environment

By bringing calculation, learning resources, symbolic solving, graph visualisation, practice, and progress tracking together, MathPilot aims to reduce the need to switch between separate tools while studying mathematics.

## How It Works

1. **Choose a learning activity:** Open the scientific calculator or select a section of the Offline Lab.
2. **Learn a concept:** Review topic explanations, formulas, and worked examples.
3. **Solve a problem:** Enter a supported mathematical expression or equation to explore its symbolic solution.
4. **Visualise the mathematics:** Use the Explore section to inspect functions and derivative plots.
5. **Practise:** Attempt questions, use hints, and check answers with the available explanations.
6. **Review progress:** Examine stored practice attempts, topic accuracy, and revision suggestions.
7. **Continue learning:** Use the results of previous practice to guide further revision.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic and mathematical operations |
| Streamlit | Interactive web application interface |
| SymPy | Symbolic mathematics and equation solving |
| NumPy | Numerical computation and array operations |
| Pandas | Data handling and CSV export |
| Plotly | Interactive mathematical visualisations, where used |
| SQLite | Local storage of practice attempts and progress |

## Architecture and Data Storage

MathPilot runs as a Python application with a Streamlit interface. Its mathematical operations are handled using the relevant computation libraries, while the learning interface connects the calculator, learning resources, solving tools, visualisations, and practice activities.

SQLite is used to store local practice history and progress in a database file named `mathpilot.db`. Practice records can be reviewed within the application and exported to CSV where supported.

The core Offline Lab is designed to process supported mathematical operations locally. It does not require a cloud-hosted AI model for its standard symbolic mathematics functions.

## Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Git, if cloning the repository.
- A compatible web browser.
- The project source files and required Python packages.

### Step 1: Clone the Repository

```bash
git clone https://github.com/vismaya0808-arch/math_scientific_calc.git
```

### Step 2: Open the Project Folder

```bash
cd math_scientific_calc
```

### Step 3: Install Dependencies

If the repository contains a `requirements.txt` file, install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

Otherwise, install the packages required by the current version of the application. For the core stack, this may include:

```bash
python -m pip install streamlit sympy numpy pandas plotly
```

### Step 4: Run the Application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL that can be opened in a web browser.

**Note:** Keep the application's Python files and supporting modules in their expected project locations. Install any additional dependencies required by optional features.

## Running MathPilot Offline

1. Install Python and the required project dependencies while internet access is available.
2. Download or clone the project source code.
3. Launch the application locally using Streamlit.
4. Use the core Offline Lab features without continuous internet connectivity, subject to the installed dependencies and local environment.

Optional online services and features that rely on external APIs may not work offline.

## Demo and Project Links

- **Source Code:** [MathPilot on GitHub](https://github.com/vismaya0808-arch/math_scientific_calc)
- **Live Application:** [Open MathPilot](https://mathscientificcalc-p6vkenjztxsrebgxnku2um.streamlit.app/)

## Data and AI Usage

MathPilot's core mathematical operations use programmatic computation through Python libraries such as SymPy and NumPy. Its local progress tracking uses SQLite, and Pandas can be used for data handling and CSV export.

The standard symbolic mathematics workflow does not require an AI model to generate mathematical results. Any optional AI-powered assistance or OCR functionality should be documented separately, including the specific model or service used, the information sent to it, and whether an internet connection is required.

The application stores practice history locally in its SQLite database. Users should consult the implementation and any enabled integrations to understand the handling of data by optional external services.

## AI Tools and Reused Code

The project team should document the development tools and external resources actually used to build MathPilot.

- **AI development tools:** List the AI assistants used during development and describe their actual role, such as brainstorming, debugging, code assistance, or documentation.
- **Libraries and dependencies:** Streamlit, SymPy, NumPy, Pandas, Plotly, and other installed packages are third-party dependencies. They should be distinguished from externally copied source code.
- **Reused code:** Identify any external code snippets, templates, or repositories incorporated into the project and provide attribution where required.
- **Team contributions:** Describe the features and code designed, implemented, integrated, or tested by the team.

This section should be updated to reflect the actual development process accurately.

## Limitations

- The supported mathematical operations depend on the expressions and input formats handled by the application.
- Symbolic results may require interpretation and are not a substitute for checking mathematical reasoning.
- Practice quality and progress insights depend on the questions, answer-checking logic, and topics currently implemented.
- Local progress is stored in the application's SQLite database and may not automatically synchronise across devices.
- Optional online AI or OCR features may require internet connectivity and additional configuration.
- The application requires its Python environment and dependencies to be installed before local offline use.

## Future Improvements

Potential improvements include expanding topic coverage, adding more detailed mathematical explanations, introducing adaptive practice based on student performance, improving accessibility, and conducting structured testing with student users.

Further work could also improve input handling, broaden the range of supported equations, and provide more detailed feedback on common mathematical mistakes.

## Team

**Team Name:** DIVINE EDGE

**Project Name:** MathPilot — Scientific Learning Studio

**Institution:** GSSS Institute of Engineering and Technology for Women, Mysuru

**Team Members:**
- VISMAYA S
- VAISHNAVI D
- SRUSHTI L
- DHANYATHA KN

---

*MathPilot aims to make mathematics more interactive by bringing calculation, exploration, practice, and revision together in one learning environment.*
