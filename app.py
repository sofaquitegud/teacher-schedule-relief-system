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
import xml.etree.ElementTree as ET
import tempfile
import os
from collections import defaultdict
from itertools import groupby
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns

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

# Updated time frames to reflect real school schedule with breaks and prayer
TIME_FRAMES = [
    "8:00-8:30",
    "8:30-9:00",
    "9:00-9:30",
    "9:30-10:00",
    "10:00-10:30",
    "10:30-11:00",
    "11:00-11:30",
    "11:30-12:00",
    "12:00-12:30",
    "12:30-1:00 (Prayer/Lunch)",  # Prayer/lunch break
    "1:00-2:00 (Prayer/Lunch)",  # Prayer/lunch break
    "2:00-2:30",
    "2:30-3:00",
    "3:00-3:30",
]

# --- Class Location and School Structure ---
FORMS = [f"Form {i}" for i in range(1, 6)]
CLASS_NAMES = ["Amanah", "Bestari", "Cekal", "Dedikasi", "Efisien", "Fitrah"]
BLOCKS = ["Block A", "Block B", "Block C"]
FLOORS = ["Ground Floor", "1st Floor", "2nd Floor", "3rd Floor"]
SPECIAL_ROOMS = [
    "Computer Lab",
    "Physics Lab",
    "Chemistry Lab",
    "Biology Lab",
    "Library",
]

# Generate class location mapping
CLASS_LOCATIONS = {}
for i, form in enumerate(FORMS):
    for j, class_name in enumerate(CLASS_NAMES):
        block = BLOCKS[(i + j) % len(BLOCKS)]
        floor = FLOORS[(j) % len(FLOORS)]
        CLASS_LOCATIONS[f"{form} {class_name}"] = {
            "block": block,
            "floor": floor,
        }
# Add special rooms
for idx, room in enumerate(SPECIAL_ROOMS):
    block = BLOCKS[idx % len(BLOCKS)]
    floor = FLOORS[(idx + 1) % len(FLOORS)]
    CLASS_LOCATIONS[room] = {"block": block, "floor": floor}


def load_sample_data():
    """Load sample teacher data with realistic consecutive teaching blocks and breaks between subjects."""
    sample_teachers = [
        {
            "id": 1,
            "name": "Cikgu Aiman",
            "subjects": ["Math", "Add Maths"],
            "free_periods": [
                "9:00-9:30",
                "11:00-11:30",
                "12:30-1:00 (Prayer/Lunch)",
                "1:00-2:00 (Prayer/Lunch)",
                "2:30-3:00",
                "3:00-3:30",
            ],
            "constraints": "",
            "relief_count": 0,
            "class_locations": {
                # Math: 8:00-9:00 (2 periods)
                "8:00-8:30": "Form 1 Amanah",
                "8:30-9:00": "Form 1 Amanah",
                # Add Maths: 10:00-11:00 (2 periods)
                "10:00-10:30": "Form 1 Bestari",
                "10:30-11:00": "Form 1 Bestari",
            },
        },
        {
            "id": 2,
            "name": "Cikgu Bella",
            "subjects": ["English"],
            "free_periods": [
                "9:00-9:30",
                "11:00-11:30",
                "12:30-1:00 (Prayer/Lunch)",
                "1:00-2:00 (Prayer/Lunch)",
                "2:30-3:00",
                "3:00-3:30",
            ],
            "constraints": "pregnant",
            "relief_count": 0,
            "class_locations": {
                # English: 8:00-9:00 (2 periods)
                "8:00-8:30": "Form 2 Amanah",
                "8:30-9:00": "Form 2 Amanah",
                # English: 10:00-11:00 (2 periods)
                "10:00-10:30": "Form 2 Bestari",
                "10:30-11:00": "Form 2 Bestari",
            },
        },
        {
            "id": 3,
            "name": "Cikgu Chan",
            "subjects": ["Science", "Biology"],
            "free_periods": [
                "9:00-9:30",
                "11:00-11:30",
                "12:30-1:00 (Prayer/Lunch)",
                "1:00-2:00 (Prayer/Lunch)",
                "2:30-3:00",
                "3:00-3:30",
            ],
            "constraints": "no_upstairs",
            "relief_count": 0,
            "class_locations": {
                # Science: 8:00-9:00 (2 periods)
                "8:00-8:30": "Form 3 Amanah",
                "8:30-9:00": "Form 3 Amanah",
                # Biology: 10:00-11:00 (2 periods)
                "10:00-10:30": "Form 3 Bestari",
                "10:30-11:00": "Form 3 Bestari",
            },
        },
        {
            "id": 4,
            "name": "Cikgu Diana",
            "subjects": ["Sejarah", "Bahasa Melayu"],
            "free_periods": [
                "9:00-9:30",
                "11:00-11:30",
                "12:30-1:00 (Prayer/Lunch)",
                "1:00-2:00 (Prayer/Lunch)",
                "2:30-3:00",
                "3:00-3:30",
            ],
            "constraints": "MC",
            "relief_count": 0,
            "class_locations": {
                # Sejarah: 8:00-9:00 (2 periods)
                "8:00-8:30": "Form 4 Amanah",
                "8:30-9:00": "Form 4 Amanah",
                # Bahasa Melayu: 10:00-11:00 (2 periods)
                "10:00-10:30": "Form 4 Bestari",
                "10:30-11:00": "Form 4 Bestari",
            },
        },
        {
            "id": 5,
            "name": "Cikgu Elly",
            "subjects": ["Physics"],
            "free_periods": [
                "8:00-8:30",
                "9:00-9:30",
                "10:30-11:00",
                "12:30-1:00 (Prayer/Lunch)",
                "1:00-2:00 (Prayer/Lunch)",
                "2:30-3:00",
                "3:00-3:30",
            ],
            "constraints": "",
            "relief_count": 0,
            "class_locations": {
                # Physics: 8:30-9:30 (2 periods)
                "8:30-9:00": "Form 5 Amanah",
                "9:00-9:30": "Form 5 Amanah",
                # Physics: 10:00-11:00 (2 periods)
                "10:00-10:30": "Form 5 Bestari",
                "11:00-11:30": "Form 5 Bestari",
            },
        },
        {
            "id": 6,
            "name": "Cikgu Farid",
            "subjects": ["ICT", "Computer Science"],
            "free_periods": [
                "8:00-8:30",
                "9:00-9:30",
                "10:30-11:00",
                "12:30-1:00 (Prayer/Lunch)",
                "1:00-2:00 (Prayer/Lunch)",
                "2:30-3:00",
                "3:00-3:30",
            ],
            "constraints": "",
            "relief_count": 0,
            "class_locations": {
                # ICT: 8:30-9:30 (2 periods)
                "8:30-9:00": "Computer Lab",
                "9:00-9:30": "Computer Lab",
                # Computer Science: 10:00-11:00 (2 periods)
                "10:00-10:30": "Form 3 Efisien",
                "11:00-11:30": "Form 3 Efisien",
            },
        },
        {
            "id": 7,
            "name": "Cikgu Hana",
            "subjects": ["Pendidikan Seni Visual"],
            "free_periods": [
                "8:00-8:30",
                "9:00-9:30",
                "10:30-11:00",
                "12:30-1:00 (Prayer/Lunch)",
                "1:00-2:00 (Prayer/Lunch)",
                "2:30-3:00",
                "3:00-3:30",
            ],
            "constraints": "",
            "relief_count": 0,
            "class_locations": {
                # PSV: 8:30-9:30 (2 periods)
                "8:30-9:00": "Form 5 Efisien",
                "9:00-9:30": "Form 5 Efisien",
                # PSV: 10:00-11:00 (2 periods)
                "10:00-10:30": "Form 5 Fitrah",
                "11:00-11:30": "Form 5 Fitrah",
            },
        },
        {
            "id": 8,
            "name": "Cikgu Izzat",
            "subjects": ["Pendidikan Jasmani"],
            "free_periods": [
                "8:00-8:30",
                "9:30-10:00",
                "10:30-11:00",
                "12:30-1:00 (Prayer/Lunch)",
                "1:00-2:00 (Prayer/Lunch)",
                "2:30-3:00",
                "3:00-3:30",
            ],
            "constraints": "",
            "relief_count": 0,
            "class_locations": {
                # PJ: 8:30-9:30 (2 periods)
                "8:30-9:00": "Form 3 Efisien",
                "9:00-9:30": "Form 3 Efisien",
                # PJ: 10:00-11:00 (2 periods)
                "10:00-10:30": "Form 3 Fitrah",
                "11:00-11:30": "Form 3 Fitrah",
            },
        },
        {
            "id": 9,
            "name": "Cikgu Julia",
            "subjects": ["Ekonomi", "Prinsip Perakaunan"],
            "free_periods": [
                "8:00-8:30",
                "9:30-10:00",
                "10:30-11:00",
                "12:30-1:00 (Prayer/Lunch)",
                "1:00-2:00 (Prayer/Lunch)",
                "2:30-3:00",
                "3:00-3:30",
            ],
            "constraints": "",
            "relief_count": 0,
            "class_locations": {
                # Ekonomi: 8:30-9:30 (2 periods)
                "8:30-9:00": "Form 5 Amanah",
                "9:00-9:30": "Form 5 Amanah",
                # Prinsip Perakaunan: 10:00-11:00 (2 periods)
                "10:00-10:30": "Form 5 Bestari",
                "11:00-11:30": "Form 5 Bestari",
            },
        },
        {
            "id": 10,
            "name": "Cikgu Kamal",
            "subjects": ["Geografi"],
            "free_periods": [
                "8:00-8:30",
                "9:00-9:30",
                "10:30-11:00",
                "12:30-1:00 (Prayer/Lunch)",
                "1:00-2:00 (Prayer/Lunch)",
                "2:30-3:00",
                "3:00-3:30",
            ],
            "constraints": "",
            "relief_count": 0,
            "class_locations": {
                # Geografi: 8:30-9:30 (2 periods)
                "8:30-9:00": "Form 1 Efisien",
                "9:00-9:30": "Form 1 Efisien",
                # Geografi: 10:00-11:00 (2 periods)
                "10:00-10:30": "Form 1 Fitrah",
                "11:00-11:30": "Form 1 Fitrah",
            },
        },
        {
            "id": 11,
            "name": "Cikgu Laila",
            "subjects": ["Moral", "Sivik"],
            "free_periods": [
                "8:00-8:30",
                "9:30-10:00",
                "10:30-11:00",
                "12:30-1:00 (Prayer/Lunch)",
                "1:00-2:00 (Prayer/Lunch)",
                "2:30-3:00",
                "3:00-3:30",
            ],
            "constraints": "",
            "relief_count": 0,
            "class_locations": {
                # Moral: 8:30-9:30 (2 periods)
                "8:30-9:00": "Form 3 Amanah",
                "9:00-9:30": "Form 3 Amanah",
                # Sivik: 10:00-11:00 (2 periods)
                "10:00-10:30": "Form 3 Bestari",
                "11:00-11:30": "Form 3 Bestari",
            },
        },
        # Additional teachers for constraint testing
        {
            "id": 13,
            "name": "Cikgu Relief",
            "subjects": ["Any"],
            "free_periods": [
                "8:00-8:30",
                "8:30-9:00",
                "10:00-10:30",
                "10:30-11:00",
                "9:00-9:30",
                "9:30-10:00",
                "11:00-11:30",
                "11:30-12:00",
                "12:00-12:30",
                "2:00-2:30",
                "2:30-3:00",
                "3:00-3:30",
            ],
            "constraints": "",
            "relief_count": 0,
            "class_locations": {},
        },
        {
            "id": 14,
            "name": "Cikgu Pregnant",
            "subjects": ["Math"],
            "free_periods": [
                "8:00-8:30",
                "8:30-9:00",
                "10:00-10:30",
                "10:30-11:00",
                "2:00-2:30",
                "2:30-3:00",
            ],
            "constraints": "pregnant",
            "relief_count": 0,
            "class_locations": {},
        },
        {
            "id": 15,
            "name": "Cikgu MC",
            "subjects": ["Science"],
            "free_periods": [
                "8:00-8:30",
                "8:30-9:00",
                "10:00-10:30",
                "10:30-11:00",
                "2:00-2:30",
                "2:30-3:00",
            ],
            "constraints": "MC",
            "relief_count": 0,
            "class_locations": {},
        },
        {
            "id": 16,
            "name": "Cikgu NoUpstairs",
            "subjects": ["English"],
            "free_periods": [
                "8:00-8:30",
                "8:30-9:00",
                "10:00-10:30",
                "10:30-11:00",
                "2:00-2:30",
                "2:30-3:00",
            ],
            "constraints": "no_upstairs",
            "relief_count": 0,
            "class_locations": {},
        },
        {
            "id": 17,
            "name": "Cikgu Flexible",
            "subjects": ["Math", "Science"],
            "free_periods": [
                "8:00-8:30",
                "8:30-9:00",
                "9:00-9:30",
                "9:30-10:00",
                "10:00-10:30",
                "10:30-11:00",
                "11:00-11:30",
                "11:30-12:00",
                "2:00-2:30",
                "2:30-3:00",
            ],
            "constraints": "",
            "relief_count": 0,
            "class_locations": {},
        },
        {
            "id": 18,
            "name": "Cikgu Upstairs",
            "subjects": ["Math"],
            "free_periods": ["8:00-8:30", "8:30-9:00", "10:00-10:30", "10:30-11:00"],
            "constraints": "",
            "relief_count": 0,
            "class_locations": {},
        },
        {
            "id": 19,
            "name": "Cikgu Extra",
            "subjects": ["Math", "English", "Science"],
            "free_periods": TIME_FRAMES.copy(),
            "constraints": "",
            "relief_count": 0,
            "class_locations": {},
        },
    ]
    st.session_state.teachers = sample_teachers
    st.session_state.teacher_relief_count = {
        teacher["id"]: 0 for teacher in sample_teachers
    }


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


# Add break/prayer periods and other duties exclusion in optimization
BREAK_PRAYER_PERIODS = [TIME_FRAMES[10], TIME_FRAMES[11]]  # 12:30-1:00, 1:00-2:00


def group_periods_into_blocks(periods):
    """Group a list of periods (strings) into consecutive blocks based on TIME_FRAMES order."""
    if not periods:
        return []
    # Sort periods by TIME_FRAMES order
    period_indices = sorted([TIME_FRAMES.index(p) for p in periods if p in TIME_FRAMES])
    blocks = []
    current_block = [period_indices[0]]
    for idx in period_indices[1:]:
        if idx == current_block[-1] + 1:
            current_block.append(idx)
        else:
            blocks.append([TIME_FRAMES[i] for i in current_block])
            current_block = [idx]
    blocks.append([TIME_FRAMES[i] for i in current_block])
    return blocks


def generate_schedule_with_pulp():
    """Generate relief schedule using PuLP optimization (block-based)."""
    st.session_state.relief_schedule = []
    st.session_state.optimization_results = {}

    if not st.session_state.absences or not st.session_state.teachers:
        return

    # Step 1: Group absences into blocks
    block_absences = []
    block_id_counter = 1
    for absence in st.session_state.absences:
        blocks = group_periods_into_blocks(absence["periods"])
        for block in blocks:
            block_absences.append(
                {
                    "block_id": f"{absence['id']}_block{block_id_counter}",
                    "teacher_id": absence["teacher_id"],
                    "teacher_name": absence["teacher_name"],
                    "subject": absence["subject"],
                    "period_block": block,  # list of periods
                    "assigned": False,
                    "relief_teacher": None,
                }
            )
            block_id_counter += 1

    teachers = st.session_state.teachers
    absences = block_absences

    # Helper to check if a block is on 2nd/3rd floor
    def is_upstairs(period_block, absent_teacher):
        teacher = next(
            (t for t in st.session_state.teachers if t["name"] == absent_teacher), None
        )
        if not teacher or not teacher.get("class_locations"):
            return False
        for period in period_block:
            cname = teacher["class_locations"].get(period)
            if cname:
                loc = CLASS_LOCATIONS.get(cname, {})
                if loc.get("floor") in ["2nd Floor", "3rd Floor"]:
                    return True
        return False

    # Decision variables: x[teacher_id, block_id]
    x = {}
    for teacher in teachers:
        for absence in absences:
            x[teacher["id"], absence["block_id"]] = pulp.LpVariable(
                f"assign_teacher_{teacher['id']}_to_block_{absence['block_id']}",
                cat="Binary",
            )

    # Objective function coefficients
    objective_coeff = {}
    for teacher in teachers:
        for absence in absences:
            coeff = 0
            coeff += 10
            subject_match = calculate_subject_match_score(
                teacher["subjects"], absence["subject"]
            )
            coeff += subject_match * 20
            current_reliefs = st.session_state.teacher_relief_count.get(
                teacher["id"], 0
            )
            coeff -= current_reliefs * 5
            if teacher["constraints"] == "MC":
                coeff -= 1000  # Will be hard-constrained below
            elif teacher["constraints"] == "pregnant":
                coeff -= 15
            elif teacher["constraints"] == "no_upstairs":
                # Mild penalty, but hard constraint below for upstairs
                coeff -= 5
            # Soft penalty: avoid assigning relief immediately after own class unless consecutive
            first_period = absence["period_block"][0]
            try:
                idx = TIME_FRAMES.index(first_period)
                if idx > 0:
                    prev_period = TIME_FRAMES[idx - 1]
                    if (
                        prev_period not in teacher["free_periods"]
                        and prev_period not in BREAK_PRAYER_PERIODS
                    ):
                        coeff -= 5
            except Exception:
                pass
            # SOFT PENALTY: If teacher is not free for any period in the block, add a large penalty
            for period in absence["period_block"]:
                if period not in teacher["free_periods"]:
                    coeff -= 100
                if period in BREAK_PRAYER_PERIODS:
                    coeff -= 100
                if "other_duties" in teacher and period in teacher["other_duties"]:
                    coeff -= 100
            objective_coeff[teacher["id"], absence["block_id"]] = coeff

    # Create the optimization problem
    prob = pulp.LpProblem("Teacher_Relief_Assignment_Blocks", pulp.LpMaximize)
    prob += pulp.lpSum(
        [
            objective_coeff[teacher["id"], absence["block_id"]]
            * x[teacher["id"], absence["block_id"]]
            for teacher in teachers
            for absence in absences
        ]
    )

    # Constraint 1: Each block assigned to at most one teacher
    for absence in absences:
        prob += (
            pulp.lpSum([x[teacher["id"], absence["block_id"]] for teacher in teachers])
            <= 1,
            f"Block_{absence['block_id']}_max_one_teacher",
        )

    # Constraint 2: MC teachers cannot be assigned (hard constraint)
    for teacher in teachers:
        if teacher["constraints"] == "MC":
            for absence in absences:
                prob += (
                    x[teacher["id"], absence["block_id"]] == 0,
                    f"Teacher_{teacher['id']}_MC_exemption_block_{absence['block_id']}",
                )

    # Constraint 3: Pregnant teachers - limited assignment (hard constraint)
    for teacher in teachers:
        if teacher["constraints"] == "pregnant":
            prob += (
                pulp.lpSum(
                    [x[teacher["id"], absence["block_id"]] for absence in absences]
                )
                <= 1,
                f"Teacher_{teacher['id']}_pregnant_limit",
            )

    # Constraint 4: NoUpstairs teachers cannot be assigned to 2nd/3rd floor (hard constraint)
    for teacher in teachers:
        if teacher["constraints"] == "no_upstairs":
            for absence in absences:
                if is_upstairs(absence["period_block"], absence["teacher_name"]):
                    prob += (
                        x[teacher["id"], absence["block_id"]] == 0,
                        f"Teacher_{teacher['id']}_no_upstairs_block_{absence['block_id']}",
                    )

    # Constraint 5: Teacher cannot cover their own absence
    for teacher in teachers:
        for absence in absences:
            if teacher["id"] == absence["teacher_id"]:
                prob += (
                    x[teacher["id"], absence["block_id"]] == 0,
                    f"Teacher_{teacher['id']}_cannot_cover_own_block_{absence['block_id']}",
                )

    # Constraint 6: Fair distribution - limit maximum assignments per teacher
    max_assignments_per_teacher = max(1, len(absences) // len(teachers) + 1)
    for teacher in teachers:
        prob += (
            pulp.lpSum([x[teacher["id"], absence["block_id"]] for absence in absences])
            <= max_assignments_per_teacher,
            f"Teacher_{teacher['id']}_max_assignments",
        )

    # Solve problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    st.session_state.optimization_results = {
        "status": pulp.LpStatus[prob.status],
        "objective_value": pulp.value(prob.objective),
        "num_variables": len(x),
        "num_constraints": len(prob.constraints),
    }

    # Extract solution and create relief schedule
    for teacher in teachers:
        for absence in absences:
            if pulp.value(x[teacher["id"], absence["block_id"]]) == 1:
                relief_assignment = {
                    "block_id": absence["block_id"],
                    "absent_teacher": absence["teacher_name"],
                    "subject": absence["subject"],
                    "period_block": absence["period_block"],
                    "relief_teacher": teacher["name"],
                    "relief_teacher_id": teacher["id"],
                    "status": "assigned",
                    "assignment_score": objective_coeff[
                        teacher["id"], absence["block_id"]
                    ],
                }
                st.session_state.relief_schedule.append(relief_assignment)
                teacher["relief_count"] += 1
                st.session_state.teacher_relief_count[teacher["id"]] += 1

    # Add unassigned blocks
    for absence in absences:
        assigned = any(
            r["block_id"] == absence["block_id"] and r["status"] == "assigned"
            for r in st.session_state.relief_schedule
        )
        if not assigned:
            relief_assignment = {
                "block_id": absence["block_id"],
                "absent_teacher": absence["teacher_name"],
                "subject": absence["subject"],
                "period_block": absence["period_block"],
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
    for relief in sorted(
        st.session_state.relief_schedule, key=lambda x: x["period_block"]
    ):
        data.append(
            [
                str(relief["period_block"]),
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


def import_asc_timetables_data(uploaded_file):
    """Import teacher schedules from ASC Timetables export file (XML or Excel)"""
    teachers_data = []

    try:
        file_ext = uploaded_file.name.split(".")[-1].lower()

        if file_ext == "xml":
            # Process XML file
            tree = ET.parse(uploaded_file)
            root = tree.getroot()

            for teacher in root.findall(
                ".//Teacher"
            ):  # Adjust xpath based on actual XML structure
                teacher_data = {
                    "id": len(st.session_state.teachers) + 1,
                    "name": teacher.get("name", ""),
                    "subjects": [],
                    "free_periods": [],
                    "constraints": "",
                    "relief_count": 0,
                    "other_duties": [],
                }

                # Extract subjects
                for subject in teacher.findall(".//Subject"):
                    teacher_data["subjects"].append(subject.get("name", ""))

                # Extract free periods by checking periods without lessons
                all_periods = set(TIME_FRAMES)
                busy_periods = set()
                for lesson in teacher.findall(".//Lesson"):
                    period = lesson.get(
                        "period", ""
                    )  # Adjust based on actual XML structure
                    if period in TIME_FRAMES:
                        busy_periods.add(period)

                teacher_data["free_periods"] = list(all_periods - busy_periods)
                teachers_data.append(teacher_data)

        elif file_ext in ["xlsx", "xls"]:
            # Process Excel file
            df = pd.read_excel(uploaded_file)

            # Assuming Excel structure has columns: Teacher, Subject, Period
            for teacher_name in df["Teacher"].unique():
                subjects_series = df[df["Teacher"] == teacher_name]["Subject"]
                teacher_data = {
                    "id": len(st.session_state.teachers) + 1,
                    "name": teacher_name,
                    "subjects": list(np.unique(subjects_series)),
                    "free_periods": [],
                    "constraints": "",
                    "relief_count": 0,
                    "other_duties": [],
                }

                # Calculate free periods
                busy_periods = set(df[df["Teacher"] == teacher_name]["Period"].tolist())
                all_periods = set(TIME_FRAMES)
                teacher_data["free_periods"] = list(all_periods - busy_periods)

                teachers_data.append(teacher_data)

        if teachers_data:
            # Update session state with imported teachers
            st.session_state.teachers.extend(teachers_data)
            st.session_state.teacher_relief_count.update(
                {teacher["id"]: 0 for teacher in teachers_data}
            )

            return len(teachers_data)
        else:
            st.warning("No teacher data found in the uploaded file.")
            return 0

    except Exception as e:
        st.error(f"Error importing data: {str(e)}")
        return 0


def create_school_map():
    """Create a visual representation of the school map with blocks, floors, and class locations."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Color scheme
    colors_scheme = {
        "Block A": "#FF6B6B",  # Red
        "Block B": "#4ECDC4",  # Teal
        "Block C": "#45B7D1",  # Blue
        "Special": "#96CEB4",  # Green
    }

    # Floor positions (y-axis)
    floor_positions = {
        "Ground Floor": 0,
        "1st Floor": 1,
        "2nd Floor": 2,
        "3rd Floor": 3,
    }

    # Block positions (x-axis)
    block_positions = {"Block A": 0, "Block B": 1, "Block C": 2}

    # Collect classes for each (block, floor)
    cell_classes = {
        (block, floor): [] for block in block_positions for floor in floor_positions
    }
    for class_name, location in CLASS_LOCATIONS.items():
        block = location["block"]
        floor = location["floor"]
        if block in block_positions and floor in floor_positions:
            cell_classes[(block, floor)].append(class_name)

    # Draw floor lines
    for floor, y in floor_positions.items():
        ax.axhline(y=y - 0.4, color="gray", alpha=0.3, linestyle="-", linewidth=1)
        ax.text(-0.5, y, floor, fontsize=10, ha="right", va="center", fontweight="bold")

    # Draw block lines
    for block, x in block_positions.items():
        ax.axvline(x=x - 0.4, color="gray", alpha=0.3, linestyle="-", linewidth=1)
        ax.text(
            x,
            -0.5,
            block,
            fontsize=10,
            ha="center",
            va="top",
            fontweight="bold",
            rotation=0,
        )

    # Plot classes in a grid within each cell
    class_count = 0
    for (block, floor), class_list in cell_classes.items():
        x = block_positions[block]
        y = floor_positions[floor]
        n = len(class_list)
        if n == 0:
            continue
        # Arrange in 2 columns, up to 3 rows (for up to 6 classes)
        cols = 2
        rows = (n + 1) // 2 if n > 2 else 1 if n == 1 else 2
        for idx, class_name in enumerate(class_list):
            row = idx // cols
            col = idx % cols
            # Offset within cell
            x_offset = (col - 0.5) * 0.35 if cols > 1 else 0
            y_offset = (0.5 - row) * 0.25 if rows > 1 else 0
            # Determine color
            if "Lab" in class_name or class_name == "Library":
                color = colors_scheme["Special"]
            else:
                color = colors_scheme[block]
            # Draw mini-rectangle
            rect = FancyBboxPatch(
                (x + x_offset - 0.15, y + y_offset - 0.10),
                0.3,
                0.2,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="black",
                linewidth=1,
                alpha=0.8,
            )
            ax.add_patch(rect)
            # Add class name
            ax.text(
                x + x_offset,
                y + y_offset,
                class_name.replace("Form ", ""),
                fontsize=8,
                ha="center",
                va="center",
                fontweight="bold",
            )
            class_count += 1

    # Set up the plot
    ax.set_xlim(-0.8, 2.8)
    ax.set_ylim(-0.8, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add title
    ax.text(
        1,
        3.5,
        "School Campus Map",
        fontsize=16,
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Add legend
    legend_elements = [
        patches.Patch(color=colors_scheme["Block A"], label="Block A"),
        patches.Patch(color=colors_scheme["Block B"], label="Block B"),
        patches.Patch(color=colors_scheme["Block C"], label="Block C"),
        patches.Patch(color=colors_scheme["Special"], label="Special Rooms"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    # Add statistics
    ax.text(
        1, -0.6, f"Total Classes: {class_count}", fontsize=10, ha="center", va="center"
    )

    plt.tight_layout()
    return fig


def download_school_map():
    """Create a downloadable version of the school map."""
    fig = create_school_map()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
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

        # Add file uploader for ASC Timetables data
        st.subheader("Import from ASC Timetables")
        uploaded_file = st.file_uploader(
            "Upload ASC Timetables export (XML or Excel)",
            type=["xml", "xlsx", "xls"],
            help="Upload your exported timetable data from ASC Timetables.",
        )

        if uploaded_file is not None:
            if st.button("Import Timetable Data"):
                with st.spinner("Importing timetable data..."):
                    num_imported = import_asc_timetables_data(uploaded_file)
                    if num_imported > 0:
                        st.success(f"Successfully imported {num_imported} teachers!")
                        st.rerun()

        if st.button("📚 Load Sample Data", type="primary"):
            load_sample_data()
            st.success("Sample data loaded!")
            st.rerun()

        # --- School Map Section ---
        st.markdown("---")
        st.subheader("🏫 School Campus Map")

        # Display the map
        fig = create_school_map()
        st.pyplot(fig)

        # Download option
        map_buffer = download_school_map()
        st.download_button(
            label="📥 Download School Map",
            data=map_buffer,
            file_name="school-campus-map.png",
            mime="image/png",
        )

        # --- Class Location Preview ---
        st.subheader("🏫 Class Locations Preview")
        for form in FORMS:
            with st.expander(form):
                for class_name in CLASS_NAMES:
                    cname = f"{form} {class_name}"
                    loc = CLASS_LOCATIONS.get(cname, {})
                    st.markdown(
                        f"**{class_name}**: {loc.get('block', '-')}, {loc.get('floor', '-')}"
                    )
        st.markdown("---")
        st.subheader("Special Rooms")
        for room in SPECIAL_ROOMS:
            loc = CLASS_LOCATIONS.get(room, {})
            st.markdown(f"**{room}**: {loc.get('block', '-')}, {loc.get('floor', '-')}")

    # Display current teachers as a table
    if st.session_state.teachers:
        st.subheader("Current Teachers")
        teachers_data = []
        for teacher in st.session_state.teachers:
            teachers_data.append(
                {
                    "Name": teacher["name"],
                    "Subjects": ", ".join(teacher["subjects"]),
                    "Constraints": teacher["constraints"],
                    "Free Periods": ", ".join(map(str, teacher["free_periods"])),
                }
            )
        teachers_df = pd.DataFrame(teachers_data)
        st.dataframe(teachers_df, use_container_width=True)

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
                selected_teacher_label = st.selectbox(
                    "Absent Teacher", list(teacher_options.keys())
                )
                selected_teacher_id = teacher_options[selected_teacher_label]
                selected_teacher = next(
                    t
                    for t in st.session_state.teachers
                    if t["id"] == selected_teacher_id
                )
                selected_subjects = st.multiselect(
                    "Subject", all_subjects, max_selections=1
                )
                # Auto-select periods where the teacher is scheduled to teach the selected subject
                auto_periods = []
                if selected_subjects:
                    subject = selected_subjects[0]
                    for period, class_name in selected_teacher.get(
                        "class_locations", {}
                    ).items():
                        if subject in selected_teacher["subjects"]:
                            auto_periods.append(period)
                # Show the periods as read-only info
                if selected_subjects:
                    st.markdown(
                        f"**Affected Periods:** {', '.join(auto_periods) if auto_periods else 'None'}"
                    )

                submitted = st.form_submit_button("Add Absence")
                if (
                    submitted
                    and selected_teacher_label
                    and selected_subjects
                    and auto_periods
                ):
                    teacher_id = selected_teacher_id
                    subject = selected_subjects[0]  # Only one subject allowed
                    add_absence(teacher_id, subject, auto_periods)
                    st.success("Absence Added!")
                    st.rerun()
        else:
            st.warning("Please add teachers first or load sample data.")

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

    with col2:
        st.header("Today's Absences")
        if st.session_state.absences:
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
        else:
            st.info("No absences recorded today.")

    # Relief Schedule Results
    if st.session_state.relief_schedule:
        st.header("\U0001f4cb Optimized Relief Schedule")

        # Helper to get class location for a block
        def get_class_location(absent_teacher, period_block):
            teacher = next(
                (t for t in st.session_state.teachers if t["name"] == absent_teacher),
                None,
            )
            if not teacher or not teacher.get("class_locations"):
                return "-"
            # Get all unique locations for the block
            locations = set()
            for period in period_block:
                cname = teacher["class_locations"].get(period)
                if cname:
                    loc = CLASS_LOCATIONS.get(cname, {})
                    locations.add(
                        f"{cname} ({loc.get('block', '-')}, {loc.get('floor', '-')})"
                    )
            return ", ".join(locations) if locations else "-"

        df = pd.DataFrame(st.session_state.relief_schedule)
        df["period_order"] = df["period_block"].apply(
            lambda x: TIME_FRAMES.index(x[0]) if x[0] in TIME_FRAMES else -1
        )
        df = df.sort_values("period_order")

        # Add class location column
        df["class_location"] = df.apply(
            lambda row: get_class_location(row["absent_teacher"], row["period_block"]),
            axis=1,
        )

        display_df = df[
            [
                "period_block",
                "absent_teacher",
                "subject",
                "relief_teacher",
                "status",
                "class_location",
            ]
        ].copy()
        display_df.columns = [
            "Period (Time Frame)",
            "Absent Teacher",
            "Subject",
            "Relief Teacher",
            "Status",
            "Class Location",
        ]

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

        # --- Relief Assignment Report ---
        st.subheader("\U0001f9fe Relief Assignment Report")
        # Count relief assignments per teacher (assigned only)
        relief_counts = {}
        for teacher in st.session_state.teachers:
            relief_counts[teacher["name"]] = sum(
                1
                for r in st.session_state.relief_schedule
                if r["relief_teacher"] == teacher["name"] and r["status"] == "assigned"
            )
        # Prepare DataFrame for display
        relief_report_df = pd.DataFrame(
            [
                {"Teacher": name, "Total Relief Assignments": count}
                for name, count in relief_counts.items()
            ]
        )
        relief_report_df = relief_report_df.sort_values(
            by="Total Relief Assignments", ascending=False
        )
        st.dataframe(relief_report_df, use_container_width=True)

        # --- Export Options and Simulate Notifications ---
        st.subheader("\U0001f4e4 Export & Notifications")
        col1, col2, col3 = st.columns(3)

        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name=f"relief-schedule-{schedule_date}.csv",
                mime="text/csv",
            )

        with col2:
            pdf_buffer = create_pdf_report(schedule_date)
            st.download_button(
                label="📄 Download PDF",
                data=pdf_buffer,
                file_name=f"relief-schedule-{schedule_date}.pdf",
                mime="application/pdf",
            )

        with col3:
            if st.button("📱 Simulate Notifications"):
                assigned_reliefs = [
                    r
                    for r in st.session_state.relief_schedule
                    if r["status"] == "assigned"
                ]
                if assigned_reliefs:
                    notifications = []
                    for relief in assigned_reliefs:
                        class_location = get_class_location(
                            relief["absent_teacher"], relief["period_block"]
                        )
                        notifications.append(
                            f"📚 Relief Duty Alert\n"
                            f"{relief['relief_teacher']}: You have relief duty for {relief['subject']} "
                            f"in Period {relief['period_block']}\n"
                            f"Class: {relief['absent_teacher']}'s class\n"
                            f"Class Location: {class_location}\n"
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
            """
        ### How Does the Relief Scheduler Work?

        1. **You tell the system which teachers are absent and when.**
        2. **The system looks at all available teachers and their free times.**
        3. **It tries to assign relief teachers so that:**
           - Each block of consecutive absent periods is covered by one relief teacher.
           - Teachers with medical leave or special conditions are not assigned.
           - Pregnant teachers are only assigned once per day.
           - Teachers are not given relief duties during their own classes or breaks.
           - The workload is shared fairly among all teachers.
        4. **The system finds the best possible arrangement and shows you the schedule.**

        **In short:**  
        The system automatically finds the best way to cover absent teachers, following school rules and teacher needs, so you don't have to do it by hand.
        """
        )


if __name__ == "__main__":
    main()
