import numpy as np

from ..graphical_simulator import GraphicalSimulator


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
    """

    # schools have different exam difficulties
    def sample_school():
        mu_question_mean = np.random.normal(loc=1.1, scale=0.2)
        sigma_question_mean = abs(np.random.normal(loc=0, scale=1))

        # hierarchical mu/sigma for the question difficulty standard deviation (logscale)
        mu_question_std = np.random.normal(loc=0.5, scale=0.3)
        sigma_question_std = abs(np.random.normal(loc=0, scale=0.5))

        return dict(
            mu_question_mean=mu_question_mean,
            sigma_question_mean=sigma_question_mean,
            mu_question_std=mu_question_std,
            sigma_question_std=sigma_question_std,
        )

    # exams have different question difficulties
    def sample_questions(mu_question_mean, sigma_question_mean, mu_question_std, sigma_question_std):
        # mean question difficulty for an exam
        question_mean = np.random.normal(loc=mu_question_mean, scale=sigma_question_mean)

        # standard deviation of question difficulty
        log_question_std = np.random.normal(loc=mu_question_std, scale=sigma_question_std)
        question_std = float(np.exp(log_question_std))

        question_difficulty = np.random.normal(loc=question_mean, scale=question_std)

        return dict(question_mean=question_mean, question_std=question_std, question_difficulty=question_difficulty)

    # realizations of individual student abilities
    def sample_student():
        student_ability = np.random.normal(loc=0, scale=1)

        return dict(student_ability=student_ability)

    # realizations of individual observations
    def sample_observation(question_difficulty, student_ability):
        theta = np.exp(question_difficulty + student_ability) / (1 + np.exp(question_difficulty + student_ability))

        obs = np.random.binomial(n=1, p=theta)

        return dict(obs=obs)

    def meta_fn():
        return {
            "num_questions": 15,  # must be constant because node is not amortizable.
            "num_students": np.random.randint(100, 201),
        }

    simulator = GraphicalSimulator(meta_fn=meta_fn)

    simulator.add_node("schools", sample_fn=sample_school)
    simulator.add_node("questions", sample_fn=sample_questions, reps="num_questions")
    simulator.add_node("students", sample_fn=sample_student, reps="num_students")
    simulator.add_node("observations", sample_fn=sample_observation)

    simulator.add_edge("schools", "questions")
    simulator.add_edge("schools", "students")
    simulator.add_edge("questions", "observations")
    simulator.add_edge("students", "observations")

    return simulator
