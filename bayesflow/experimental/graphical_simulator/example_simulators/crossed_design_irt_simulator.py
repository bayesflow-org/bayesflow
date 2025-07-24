import numpy as np

from ..graphical_simulator import GraphicalSimulator


def crossed_design_irt_simulator():
    r"""
    Item Response Theory (IRT) model implemented as a graphical simulator.

      schools
       /     \
    exams  students
      |       |
    questions |
       \     /
     observations
    """

    # schools have different exam difficulties
    def sample_school():
        mu_exam_mean = np.random.normal(loc=1.1, scale=0.2)
        sigma_exam_mean = abs(np.random.normal(loc=0, scale=1))

        # hierarchical mu/sigma for the exam difficulty standard deviation (logscale)
        mu_exam_std = np.random.normal(loc=0.5, scale=0.3)
        sigma_exam_std = abs(np.random.normal(loc=0, scale=0.5))

        return dict(
            mu_exam_mean=mu_exam_mean,
            sigma_exam_mean=sigma_exam_mean,
            mu_exam_std=mu_exam_std,
            sigma_exam_std=sigma_exam_std,
        )

    # exams have different question difficulties
    def sample_exam(mu_exam_mean, sigma_exam_mean, mu_exam_std, sigma_exam_std):
        # mean question difficulty for an exam
        exam_mean = np.random.normal(loc=mu_exam_mean, scale=sigma_exam_mean)

        # standard deviation of question difficulty
        log_exam_std = np.random.normal(loc=mu_exam_std, scale=sigma_exam_std)
        exam_std = float(np.exp(log_exam_std))

        return dict(exam_mean=exam_mean, exam_std=exam_std)

    # realizations of individual question difficulties
    def sample_question(exam_mean, exam_std):
        question_difficulty = np.random.normal(loc=exam_mean, scale=exam_std)

        return dict(question_difficulty=question_difficulty)

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
            "num_exams": np.random.randint(2, 4),
            "num_questions": np.random.randint(10, 21),
            "num_students": np.random.randint(100, 201),
        }

    simulator = GraphicalSimulator(meta_fn=meta_fn)

    simulator.add_node("schools", sample_fn=sample_school)
    simulator.add_node("exams", sample_fn=sample_exam, reps="num_exams")
    simulator.add_node("questions", sample_fn=sample_question, reps="num_questions")
    simulator.add_node("students", sample_fn=sample_student, reps="num_students")
    simulator.add_node("observations", sample_fn=sample_observation)

    simulator.add_edge("schools", "exams")
    simulator.add_edge("schools", "students")
    simulator.add_edge("exams", "questions")
    simulator.add_edge("questions", "observations")
    simulator.add_edge("students", "observations")

    return simulator
