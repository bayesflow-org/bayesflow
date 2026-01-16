import numpy as np

from ..graphical_simulator import GraphicalSimulator


def sample_schools():
    school_mu = np.random.normal()
    school_sigma = np.abs(np.random.normal())

    return {"school_mu": school_mu, "school_sigma": school_sigma}


def sample_classrooms(school_mu, school_sigma):
    classroom_mu = np.random.normal(school_mu, school_sigma)
    classroom_sigma = np.abs(np.random.normal())

    return {"classroom_mu": classroom_mu, "classroom_sigma": classroom_sigma}


def sample_students(classroom_mu, classroom_sigma):
    student_mu = np.random.normal(classroom_mu, classroom_sigma)
    student_sigma = np.abs(np.random.normal())

    return {"student_mu": student_mu, "student_sigma": student_sigma}


def sample_shared():
    shared_sigma = np.abs(np.random.normal())

    return {"shared_sigma": shared_sigma}


def sample_scores(
    student_mu,
    student_sigma,
    shared_sigma,
):
    score_mu = np.random.normal(student_mu, student_sigma)
    score_sigma = np.abs(np.random.normal(0, shared_sigma))

    y = np.random.normal(score_mu, score_sigma)

    return {"y": y}


def meta_fn():
    N_classrooms = np.random.randint(5, 30)
    N_students = np.random.randint(5, 30)
    N_scores = np.random.randint(20, 50)

    return {"N_classrooms": N_classrooms, "N_students": N_students, "N_scores": N_scores}


def three_level_simulator():
    r"""
    Simple hierarchical model with three levels of parameters:
    (test scores within students, students within classrooms, classrooms within schools)

      schools
         |
         |
    classrooms
         |
         |     shared
     students    /
          \     /
           \   /
          scores
    """
    simulator = GraphicalSimulator(meta_fn=meta_fn)

    simulator.add_node("schools", sample_fn=sample_schools)
    simulator.add_node("classrooms", sample_fn=sample_classrooms, reps="N_classrooms")
    simulator.add_node("students", sample_fn=sample_students, reps="N_students")
    simulator.add_node("shared", sample_fn=sample_shared)
    simulator.add_node("scores", sample_fn=sample_scores, reps="N_scores")

    simulator.add_edge("schools", "classrooms")
    simulator.add_edge("classrooms", "students")
    simulator.add_edge("students", "scores")
    simulator.add_edge("shared", "scores")

    return simulator
