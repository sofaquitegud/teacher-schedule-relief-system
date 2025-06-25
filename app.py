import streamlit as st
import pandas as pd
from datetime import datetime, date
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import pulp
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="AI Teacher Relief System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .teacher-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    .absence-card {
        background: #fff3cd;
        color: #212529;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
    
    .relief-assigned {
        background: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    
    .relief-unassigned {
        background: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    
    .optimization-info {
        background: #e7f3ff;
        color: #212529;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialise session state
if "teachers" not in st.session_state:
    st.session_state.teachers = []
if "absences" not in st.session_state:
    st.session_state.absences = []
if "relief_schedule" not in st.session_state:
    st.session_state.relief_schedule = []
if "teacher_relief_count" not in st.session_state:
    st.session_state.teacher_relief_count = {}
if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = {}

# Fixed list of time frames for the school day
TIME_FRAMES = [
    "8:00-8:30",
    "8:30-9:00",
    "9:00-9:30",
    "9:30-10:00",
    "10:00-10:30",
    "10:30-11:00",
    "11:00-11:30",
    "11:30-12:00",
]


def load_sample_data():
    """Load sample teacher data"""
    sample_teachers = [
        {
            "id": 1,
            "name": "Cikgu Rafiza",
            "subjects": ["Math", "Add Maths"],
            "free_periods": [TIME_FRAMES[1], TIME_FRAMES[3], TIME_FRAMES[5]],
            "constraints": "",
            "relief_count": 0,
        },
        {
            "id": 2,
            "name": "Cikgu Farid",
            "subjects": ["Chemistry", "Biology"],
            "free_periods": [TIME_FRAMES[0], TIME_FRAMES[2], TIME_FRAMES[6]],
            "constraints": "",
            "relief_count": 0,
        },
        {
            "id": 3,
            "name": "Cikgu Riza",
            "subjects": ["English"],
            "free_periods": [TIME_FRAMES[1], TIME_FRAMES[4], TIME_FRAMES[7]],
            "constraints": "pregnant",
            "relief_count": 0,
        },
        {
            "id": 4,
            "name": "Cikgu Ahmad",
            "subjects": ["Sejarah", "Bahasa Melayu"],
            "free_periods": [TIME_FRAMES[0], TIME_FRAMES[3], TIME_FRAMES[5]],
            "constraints": "",
            "relief_count": 0,
        },
        {
            "id": 5,
            "name": "Cikgu Meriani",
            "subjects": ["Prinsip Perakaunan", "Ekonomi"],
            "free_periods": [TIME_FRAMES[2], TIME_FRAMES[4], TIME_FRAMES[6]],
            "constraints": "no_upstairs",
            "relief_count": 0,
        },
        {
            "id": 6,
            "name": "Cikgu Jamal",
            "subjects": ["Grafik Komunikasi Teknikal"],
            "free_periods": [TIME_FRAMES[1], TIME_FRAMES[3], TIME_FRAMES[7]],
            "constraints": "",
            "relief_count": 0,
        },
        {
            "id": 7,
            "name": "Cikgu Nisa",
            "subjects": ["Perniagaan", "Pendidikan Seni Visual"],
            "free_periods": [TIME_FRAMES[0], TIME_FRAMES[2], TIME_FRAMES[4]],
            "constraints": "",
            "relief_count": 0,
        },
        {
            "id": 8,
            "name": "Cikgu Omar",
            "subjects": ["Computer Science", "Geografi"],
            "free_periods": [TIME_FRAMES[3], TIME_FRAMES[5], TIME_FRAMES[7]],
            "constraints": "",
            "relief_count": 0,
        },
    ]

    st.session_state.teachers = sample_teachers
    st.session_state.teacher_relief_count = {
        teacher["id"]: 0 for teacher in sample_teachers
    }


def add_teacher(name, subjects, free_periods, constraints):
    """Add new teacher"""
    teacher_id = len(st.session_state.teachers) + 1
    teacher = {
        "id": teacher_id,
        "name": name,
        "subjects": subjects,
        "free_periods": free_periods,
        "constraints": constraints,
        "relief_count": 0,
    }
    st.session_state.teachers.append(teacher)
    st.session_state.teacher_relief_count[teacher_id] = 0


def add_absence(teacher_id, subject, periods):
    """Add teacher absence for multiple periods"""
    teacher = next(
        (t for t in st.session_state.teachers if t["id"] == teacher_id), None
    )
    if teacher:
        absence = {
            "id": len(st.session_state.absences) + 1,
            "teacher_id": teacher_id,
            "teacher_name": teacher["name"],
            "subject": subject,
            "periods": periods,  # list of time frames
            "assigned": False,
            "relief_teacher": None,
        }
        st.session_state.absences.append(absence)


def calculate_subject_match_score(teacher_subjects, absence_subject):
    """Calculate subject matching score"""
    for subject in teacher_subjects:
        if (
            subject.lower() in absence_subject.lower()
            or absence_subject.lower() in subject.lower()
        ):
            return 1.0
    return 0.0


def generate_schedule_with_pulp():
    """Generate relief schedule using PuLP optimization"""
    # Clear previous result
    st.session_state.relief_schedule = []
    st.session_state.optimization_results = {}

    if not st.session_state.absences or not st.session_state.teachers:
        return

    # Expand absences: each period in an absence becomes a separate assignment need
    expanded_absences = []
    for absence in st.session_state.absences:
        for period in absence["periods"]:
            expanded_absences.append(
                {
                    "id": f"{absence['id']}_{period}",
                    "teacher_id": absence["teacher_id"],
                    "teacher_name": absence["teacher_name"],
                    "subject": absence["subject"],
                    "period": period,
                    "assigned": False,
                    "relief_teacher": None,
                }
            )

    # Create the optimization problem
    prob = pulp.LpProblem("Teacher_Relief_Assignment", pulp.LpMaximize)

    # Sets
    teachers = st.session_state.teachers
    absences = expanded_absences

    # Decision variables: x[i,j] = 1 if teacher i is assigned to absence j
    x = {}
    for teacher in teachers:
        for absence in absences:
            x[teacher["id"], absence["id"]] = pulp.LpVariable(
                f"assign_teacher_{teacher['id']}_to_absence_{absence['id']}",
                cat="Binary",
            )

    # Objective function coefficients
    objective_coeff = {}

    for teacher in teachers:
        for absence in absences:
            coeff = 0

            # Base assignment score
            coeff += 10

            # Subject match bonus
            subject_match = calculate_subject_match_score(
                teacher["subjects"], absence["subject"]
            )
            coeff += subject_match * 20

            # Fair distribution: penalize teachers with more reliefs
            current_reliefs = st.session_state.teacher_relief_count.get(
                teacher["id"], 0
            )
            coeff -= current_reliefs * 5

            # Constraint penalties
            if teacher["constraints"] == "MC":
                coeff -= 1000  # Effectively prevents assignment
            elif teacher["constraints"] == "pregnant":
                coeff -= 15  # Strong preference to avoid
            elif teacher["constraints"] == "no_upstairs":
                coeff -= 5  # Mild penalty

            objective_coeff[teacher["id"], absence["id"]] = coeff

    # Objective function: Maximize total assignment score
    prob += pulp.lpSum(
        [
            objective_coeff[teacher["id"], absence["id"]]
            * x[teacher["id"], absence["id"]]
            for teacher in teachers
            for absence in absences
        ]
    )

    # Constraints

    # Constraint 1: Each absence should be assigned to at most one teacher
    for absence in absences:
        prob += (
            pulp.lpSum([x[teacher["id"], absence["id"]] for teacher in teachers]) <= 1,
            f"Absence_{absence['id']}_max_one_teacher",
        )

    # Constraint 2: Teacher availability - can only be assigned if free during that period
    for teacher in teachers:
        for absence in absences:
            if absence["period"] not in teacher["free_periods"]:
                prob += (
                    x[teacher["id"], absence["id"]] == 0,
                    f"Teacher_{teacher['id']}_not_free_period_{absence['period']}_absence_{absence['id']}",
                )

    # Constraint 3: Teacher cannot cover their own absence
    for teacher in teachers:
        for absence in absences:
            if teacher["id"] == absence["teacher_id"]:
                prob += (
                    x[teacher["id"], absence["id"]] == 0,
                    f"Teacher_{teacher['id']}_cannot_cover_own_absence_{absence['id']}",
                )

    # Constraint 4: MC constraint - teachers with MC exemption cannot be assigned
    for teacher in teachers:
        if teacher["constraints"] == "MC":
            for absence in absences:
                prob += (
                    x[teacher["id"], absence["id"]] == 0,
                    f"Teacher_{teacher['id']}_MC_exemption_absence_{absence['id']}",
                )

    # Constraint 5: Fair distribution - limit maximum assignments per teacher
    max_assignments_per_teacher = max(1, len(absences) // len(teachers) + 1)
    for teacher in teachers:
        prob += (
            pulp.lpSum([x[teacher["id"], absence["id"]] for absence in absences])
            <= max_assignments_per_teacher,
            f"Teacher_{teacher['id']}_max_assignments",
        )

    # Constraint 6: Pregnant teachers - limited assignment
    for teacher in teachers:
        if teacher["constraints"] == "pregnant":
            prob += (
                pulp.lpSum([x[teacher["id"], absence["id"]] for absence in absences])
                <= 1,
                f"Teacher_{teacher['id']}_pregnant_limit",
            )

    # Solve problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))  # Supress solver output

    # Store optimization results
    st.session_state.optimization_results = {
        "status": pulp.LpStatus[prob.status],
        "objective_value": pulp.value(prob.objective),
        "num_variables": len(x),
        "num_constraints": len(prob.constraints),
    }

    # Extract solution and create relief schedule
    for teacher in teachers:
        for absence in absences:
            if pulp.value(x[teacher["id"], absence["id"]]) == 1:
                relief_assignment = {
                    "absence_id": absence["id"],
                    "absent_teacher": absence["teacher_name"],
                    "subject": absence["subject"],
                    "period": absence["period"],
                    "relief_teacher": teacher["name"],
                    "relief_teacher_id": teacher["id"],
                    "status": "assigned",
                    "assignment_score": objective_coeff[teacher["id"], absence["id"]],
                }

                st.session_state.relief_schedule.append(relief_assignment)

                # Update teacher relief count
                teacher["relief_count"] += 1
                st.session_state.teacher_relief_count[teacher["id"]] += 1

                # Mark absence as assigned
                # (not used in expanded absences, but kept for compatibility)

    # Add unassigned absences
    for absence in absences:
        assigned = any(
            r["absence_id"] == absence["id"] and r["status"] == "assigned"
            for r in st.session_state.relief_schedule
        )
        if not assigned:
            relief_assignment = {
                "absence_id": absence["id"],
                "absent_teacher": absence["teacher_name"],
                "subject": absence["subject"],
                "period": absence["period"],
                "relief_teacher": "UNASSIGNED",
                "relief_teacher_id": None,
                "status": "unassigned",
                "assignment_score": 0,
            }
            st.session_state.relief_schedule.append(relief_assignment)


def create_pdf_report(schedule_date):
    """Create PDF report of the relief schedule"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#667eea"),
        alignment=1,  # Center alignment
    )

    # Title
    story.append(Paragraph("Teacher Relief Schedule", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Date: {schedule_date}", styles["Normal"]))
    story.append(Paragraph(f"Generated using PuLP Optimization", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Optimization result
    if st.session_state.optimization_results:
        results = st.session_state.optimization_results
        story.append(
            Paragraph(f"Optimization Status: {results['status']}", styles["Normal"])
        )
        story.append(
            Paragraph(
                f"Objective Value: {results.get('objective_value', 'N/A')}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 12))

    # Table data
    data = [
        ["Period", "Absent Teacher", "Subject", "Relief Teacher", "Status", "Score"]
    ]
    for relief in sorted(st.session_state.relief_schedule, key=lambda x: x["period"]):
        data.append(
            [
                str(relief["period"]),
                relief["absent_teacher"],
                relief["subject"],
                relief["relief_teacher"],
                relief["status"].upper(),
                str(relief.get("assingment_score", "N/A")),
            ]
        )

    # Create table
    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, "black"),
            ]
        )
    )

    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>🎯 AI Teacher Relief Scheduler</h1>
        <p style="color: white; text-align: center; margin: 0;">Powered by PuLP Optimization Engine - Phase 1 MVP</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar for teacher management
    with st.sidebar:
        st.header("👥 Teacher Management")

        if st.button("📚 Load Sample Data", type="primary"):
            load_sample_data()
            st.success("Sample data loaded!")
            st.rerun()

        st.subheader("Add New Teacher")
        with st.form("add_teacher_form"):
            teacher_name = st.text_input("Teacher Name", key="add_teacher_name")
            subjects_input = st.multiselect(
                "Subjects (select one or more)",
                [
                    "Math",
                    "Add Maths",
                    "Chemistry",
                    "Biology",
                    "English",
                    "Sejarah",
                    "Bahasa Melayu",
                    "Prinsip Perakaunan",
                    "Ekonomi",
                    "Grafik Komunikasi Teknikal",
                    "Perniagaan",
                    "Pendidikan Seni Visual",
                    "Computer Science",
                    "Geografi",
                ],
                key="add_teacher_subjects",
            )
            periods_input = st.multiselect(
                "Free Periods (select one or more time frames)",
                TIME_FRAMES,
                key="add_teacher_periods",
            )
            constraints = st.selectbox(
                "Constraints",
                ["", "no_upstairs", "pregnant", "MC"],
                key="add_teacher_constraints",
            )

            submitted = st.form_submit_button("Add Teacher")
            if submitted and teacher_name:
                subjects = subjects_input
                free_periods = periods_input
                add_teacher(teacher_name, subjects, free_periods, constraints)
                st.success(f"Added {teacher_name}!")
                # Clear the form fields by resetting their keys
                st.session_state["add_teacher_name"] = ""
                st.session_state["add_teacher_subjects"] = []
                st.session_state["add_teacher_periods"] = []
                st.session_state["add_teacher_constraints"] = ""
                st.rerun()

    # Display current teachers
    if st.session_state.teachers:
        st.subheader("Current Teachers")
        for i, teacher in enumerate(st.session_state.teachers):
            with st.expander(
                f"{teacher['name']} (Relief Count: {teacher['relief_count']})"
            ):
                st.write(f"**Subjects:** {', '.join(teacher['subjects'])}")
                st.write(
                    f"**Free Periods:** {', '.join(map(str, teacher['free_periods']))}"
                )
                if teacher["constraints"]:
                    st.write(f"**Constraints:** {teacher['constraints']}")
                if st.button("Remove Teacher", key=f"remove_teacher_{i}"):
                    st.session_state.teachers.pop(i)
                    st.session_state.teacher_relief_count.pop(teacher["id"], None)
                    st.rerun()

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("\U0001f4da Daily Absence Input")

        schedule_date = st.date_input("Schedule Date", value=date.today())

        if st.session_state.teachers:
            # Gather all unique subjects from teachers
            all_subjects = sorted(
                {subj for t in st.session_state.teachers for subj in t["subjects"]}
            )
            with st.form("add_absence_form"):
                teacher_options = {
                    f"{t['name']} (ID: {t['id']})": t["id"]
                    for t in st.session_state.teachers
                }
                selected_teacher = st.selectbox(
                    "Absent Teacher", list(teacher_options.keys())
                )
                selected_subjects = st.multiselect(
                    "Subject", all_subjects, max_selections=1
                )
                periods = st.multiselect("Periods (Time Frames)", TIME_FRAMES)

                submitted = st.form_submit_button("Add Absence")
                if submitted and selected_teacher and selected_subjects and periods:
                    teacher_id = teacher_options[selected_teacher]
                    subject = selected_subjects[0]  # Only one subject allowed
                    add_absence(teacher_id, subject, periods)
                    st.success("Absence Added!")
                    st.rerun()
        else:
            st.warning("Please add teachers first or load sample data.")

        # Display current absences
        if st.session_state.absences:
            st.subheader("Today's Absences")
            for i, absence in enumerate(st.session_state.absences):
                st.markdown(
                    f"""
                <div class="absence-card">
                    <strong>{absence['teacher_name']}</strong> - {absence['subject']} (Periods: {', '.join(absence['periods'])})
                </div>
                """,
                    unsafe_allow_html=True,
                )

                if st.button(f"Remove", key=f"remove_absence_{i}"):
                    st.session_state.absences.pop(i)
                    st.rerun()

    with col2:
        st.header("⚡ Generate Schedule")

        if st.button(
            "🚀 Generate Relief Schedule",
            type="primary",
            disabled=len(st.session_state.absences) == 0,
        ):
            try:
                with st.spinner("Running optimization..."):
                    generate_schedule_with_pulp()
                st.success("Schedule optimized!")
                st.rerun()
            except Exception as e:
                st.warning(f"Optimization failed: {e}")
                # Optionally, you can also log the error for debugging

        # Display optimization info
        if st.session_state.optimization_results:
            st.markdown(
                """
            <div class="optimization-info">
                <h4>🔬 Optimization Results</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            results = st.session_state.optimization_results
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Status", results.get("status", "Unknown"))
                st.metric("Variables", results.get("num_variables", 0))

            with col_b:
                st.metric("Objective Value", f"{results.get('objective_value', 0):.1f}")
                st.metric("Constraints", results.get("num_constraints", 0))

        if st.session_state.relief_schedule:
            st.subheader("📊 Schedule Statistics")

            total_absences = len(st.session_state.absences)
            assigned_reliefs = sum(
                1 for r in st.session_state.relief_schedule if r["status"] == "assigned"
            )
            unassigned_reliefs = total_absences - assigned_reliefs
            teachers_used = len(
                set(
                    r["relief_teacher_id"]
                    for r in st.session_state.relief_schedule
                    if r["relief_teacher_id"]
                )
            )

            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                st.metric("Total Absences", total_absences)
            with col_b:
                st.metric("Assigned", assigned_reliefs)
            with col_c:
                st.metric("Unassigned", unassigned_reliefs)
            with col_d:
                st.metric("Teachers Used", teachers_used)

    # Relief Schedule Results
    if st.session_state.relief_schedule:
        st.header("\U0001f4cb Optimized Relief Schedule")

        # Create DataFrame for better display
        df = pd.DataFrame(st.session_state.relief_schedule)
        # Sort by time frame order
        df["period_order"] = df["period"].apply(
            lambda x: TIME_FRAMES.index(x) if x in TIME_FRAMES else -1
        )
        df = df.sort_values("period_order")

        # Display assignment scores
        display_df = df[
            [
                "period",
                "absent_teacher",
                "subject",
                "relief_teacher",
                "status",
                "assignment_score",
            ]
        ].copy()
        display_df.columns = [
            "Period (Time Frame)",
            "Absent Teacher",
            "Subject",
            "Relief Teacher",
            "Status",
            "Score",
        ]

        # Style the dataframe
        def highlight_status(val):
            if val == "assigned":
                return "background-color: #d4edda; color: #155724"
            elif val == "unassigned":
                return "background-color: #f8d7da; color: #721c24"
            return ""

        styled_df = display_df.style.apply(
            lambda col: (
                [highlight_status(v) for v in col]
                if col.name == "Status"
                else [""] * len(col)
            )
        )
        st.dataframe(styled_df, use_container_width=True)

        # Export options
        st.subheader("📤 Export Options")
        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name=f"relief-schedule-{schedule_date}.csv",
                mime="text/csv",
            )

        with col2:
            # PDF export
            pdf_buffer = create_pdf_report(schedule_date)
            st.download_button(
                label="📄 Download PDF",
                data=pdf_buffer,
                file_name=f"relief-schedule-{schedule_date}.pdf",
                mime="application/pdf",
            )

        with col3:
            # Notification simulation
            if st.button("📱 Simulate Notifications"):
                assigned_reliefs = [
                    r
                    for r in st.session_state.relief_schedule
                    if r["status"] == "assigned"
                ]
                if assigned_reliefs:
                    notifications = []
                    for relief in assigned_reliefs:
                        notifications.append(
                            f"📚 Relief Duty Alert\n"
                            f"{relief['relief_teacher']}: You have relief duty for {relief['subject']} "
                            f"in Period {relief['period']}\n"
                            f"Class: {relief['absent_teacher']}'s class\n"
                            f"Assignment Score: {relief.get('assignment_score', 'N/A')}"
                        )

                    st.success(
                        f"Notifications prepared for {len(notifications)} teachers!"
                    )
                    with st.expander("View Notification Messages"):
                        for notification in notifications:
                            st.text(notification)
                            st.markdown("---")

                else:
                    st.warning("No relief assignments to notify about.")

    # Information about the algorithm
    with st.expander("🧠 About the Algorithm"):
        st.markdown(
            f"""
        ### How the Algorithm Works (Phase 1):

        **Absence Model:**
        - Each absence entry cover multiple periods (time frames).
        - The scheduler expands each absence into individual period-based relief needs for optimization.
        - The UI shows one absence per teacher per day, but the schedule assigns relief for each selected period.

        **Objective Function:**
        - Maximizes total assignment score across all relief assignments (per period)
        - Subject match: +20 points for matching subjects
        - Base assignment: +10 points per assignment
        - Fair distribution: -5 points per existing relief count
        - Constraint penalties: Medical (-1000), Pregnant (-15), No upstairs (-5)

        **Hard Constraints:**
        1. Each period of absence assigned to at most one relief teacher
        2. Teachers only assigned during their free periods (time frames)
        3. Teachers cannot cover their own absences
        4. Medical exemption teachers cannot be assigned
        5. Fair distribution limits per teacher
        6. Pregnant teachers limited to max 1 assignment per day

        **Optimization Engine:**
        - Uses CBC (Coin-or Branch and Cut) solver
        - Binary decision variables for each teacher-absence-period pair
        - Guarantees optimal solution within constraints
        - Handles complex constraint interactions automatically
        """
        )


if __name__ == "__main__":
    main()
