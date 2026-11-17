import numpy as np
from scipy.special import expit

from ..multilevel_simulator import MultilevelSimulator


# schools have different exam difficulties
def sample_school(batch_size):
    mu_question_mean = np.random.normal(loc=1.1, scale=0.2, size=batch_size)
    sigma_question_mean = abs(np.random.normal(loc=0, scale=1, size=batch_size))

    # hierarchical mu/sigma for the question difficulty standard deviation (logscale)
    mu_question_std = np.random.normal(loc=0.5, scale=0.3, size=batch_size)
    sigma_question_std = abs(np.random.normal(loc=0, scale=0.5, size=batch_size))

    return dict(
        mu_question_mean=mu_question_mean,
        sigma_question_mean=sigma_question_mean,
        mu_question_std=mu_question_std,
        sigma_question_std=sigma_question_std,
    )


# exams have different question difficulties
def sample_questions(mu_question_mean, sigma_question_mean, mu_question_std, sigma_question_std):
    # mean question difficulty for an exam
    question_mean = np.random.normal(loc=mu_question_mean[:, 0], scale=sigma_question_mean[:, 0])

    # standard deviation of question difficulty
    log_question_std = np.random.normal(loc=mu_question_std[:, 0], scale=sigma_question_std[:, 0])
    question_std = np.exp(log_question_std)

    question_difficulty = np.random.normal(loc=question_mean, scale=question_std)

    return dict(question_mean=question_mean, question_std=question_std, question_difficulty=question_difficulty)


# realizations of individual student abilities
def sample_student(batch_size):
    student_ability = np.random.normal(loc=0, scale=1, size=batch_size)

    return dict(student_ability=student_ability)


# realizations of individual observations
def sample_observation(question_difficulty, student_ability):
    logit = question_difficulty[:, 0] + student_ability[:, 0]
    theta = expit(logit)

    obs = np.random.binomial(n=1, p=theta)

    return dict(obs=obs)


def meta_fn():
    return {
        "num_questions": 15,  # must be constant because node is not amortizable.
        "num_students": np.random.randint(100, 201),
    }


def crossed_design_irt_simulator():
    r"""
    Item Response Theory (IRT) model implemented as a graphical simulator.

      schools
       /     \
      |  students
      |       |
    questions |
       \     /
     observations

    The sample functions draw all samples of a node in one call.
    """
    simulator = MultilevelSimulator(meta_fn=meta_fn)

    simulator.add_node("schools", sample_fn=sample_school, is_batched=True)
    simulator.add_node("questions", sample_fn=sample_questions, is_batched=True, reps="num_questions")
    simulator.add_node("students", sample_fn=sample_student, is_batched=True, reps="num_students")
    simulator.add_node("observations", sample_fn=sample_observation, is_batched=True)

    simulator.add_edge("schools", "questions")
    simulator.add_edge("schools", "students")
    simulator.add_edge("questions", "observations")
    simulator.add_edge("students", "observations")

    return simulator
