# AI Teacher Schedule Relief System

## Overview

The AI Teacher Schedule Relief System is a web-based application designed to help schools efficiently manage and assign relief teachers when regular teachers are absent. Powered by optimization algorithms, the system ensures fair, constraint-aware, and practical relief scheduling, saving time and reducing manual errors.

## Features

- **Teacher Management:** Import teacher schedules from ASC Timetables (XML/Excel) or load sample data.
- **Absence Input:** Easily record daily teacher absences and affected periods.
- **Automated Relief Assignment:** Uses PuLP optimization to assign relief teachers, considering:
  - Teacher availability (free periods)
  - Special constraints (medical leave, pregnancy, no upstairs, etc.)
  - Fair distribution of relief duties
  - Class locations and school structure
- **Visual School Map:** View a graphical map of the school campus, blocks, floors, and class locations.
- **Optimized Relief Schedule:** View the generated relief schedule in a clear, sortable table.
- **Relief Assignment Report:** See a summary table of total relief assignments per teacher for the day.
- **Export Options:** Download the schedule as CSV or PDF.
- **Simulate Notifications:** Preview notification messages for assigned relief teachers.

## Installation

1. **Clone the repository:**
   ```sh
   git clone github.com/sofaquitegud/teacher-schedule-relief-system.git
   cd teacher-schedule-relief-system
   ```
2. **Install dependencies:**
   It is recommended to use a virtual environment (e.g., conda or venv).
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Run the app:**
   ```sh
   streamlit run app.py
   ```
2. **Open your browser:**
   Go to the URL shown in your terminal (usually http://localhost:8501).
3. **Workflow:**
   - Use the sidebar to import teacher data or load sample data.
   - View the school map and class locations.
   - Enter daily absences and generate the relief schedule.
   - Review the optimized schedule and summary report.
   - Export results or simulate notifications as needed.

## Credits

- Developed using [Streamlit](https://streamlit.io/), [PuLP](https://coin-or.github.io/pulp/), [Pandas](https://pandas.pydata.org/), and [ReportLab](https://www.reportlab.com/).
- School map visualization uses [matplotlib](https://matplotlib.org/).

---

For questions or contributions, please open an issue or pull request on the repository.
