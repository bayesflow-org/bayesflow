import numpy as np

from ..multilevel_simulator import MultilevelSimulator


def sample_schools(batch_size):
    school_mu = np.random.normal(size=batch_size)
    school_sigma = np.abs(np.random.normal(size=batch_size))

    return {"school_mu": school_mu, "school_sigma": school_sigma}


def sample_classrooms(school_mu, school_sigma, batch_size):
    classroom_mu = np.random.normal(school_mu[:, 0], school_sigma[:, 0])
    classroom_sigma = np.abs(np.random.normal(size=batch_size))

    return {"classroom_mu": classroom_mu, "classroom_sigma": classroom_sigma}


def sample_students(classroom_mu, classroom_sigma, batch_size):
    student_mu = np.random.normal(classroom_mu[:, 0], classroom_sigma[:, 0])
    student_sigma = np.abs(np.random.normal(size=batch_size))

    return {"student_mu": student_mu, "student_sigma": student_sigma}


def sample_shared(batch_size):
    shared_sigma = np.abs(np.random.normal(size=batch_size))

    return {"shared_sigma": shared_sigma}


def sample_scores(
    student_mu,
    student_sigma,
    shared_sigma,
):
    score_mu = np.random.normal(student_mu[:, 0], student_sigma[:, 0])
    score_sigma = np.abs(np.random.normal(0, shared_sigma[:, 0]))

    y = np.random.normal(score_mu, score_sigma)

    return {"y": y}


def meta_fn():
    N_classrooms = np.random.randint(5, 10)
    N_students = np.random.randint(5, 10)
    N_scores = np.random.randint(10, 20)

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

    The sample functions draw all samples of a node in one call.
    """
    simulator = MultilevelSimulator(meta_fn=meta_fn)

    simulator.add_node("schools", sample_fn=sample_schools, is_batched=True)
    simulator.add_node("classrooms", sample_fn=sample_classrooms, is_batched=True, reps="N_classrooms")
    simulator.add_node("students", sample_fn=sample_students, is_batched=True, reps="N_students")
    simulator.add_node("shared", sample_fn=sample_shared, is_batched=True)
    simulator.add_node("scores", sample_fn=sample_scores, is_batched=True, reps="N_scores")

    simulator.add_edge("schools", "classrooms")
    simulator.add_edge("classrooms", "students")
    simulator.add_edge("students", "scores")
    simulator.add_edge("shared", "scores")

    return simulator
