import pytest


@pytest.fixture()
def diagonal_normal():
    from bayesflow.distributions import DiagonalNormal

    return DiagonalNormal(mean=1.0, std=2.0)


@pytest.fixture()
def diagonal_student_t():
    from bayesflow.distributions import DiagonalStudentT

    return DiagonalStudentT(df=10, loc=1.0, scale=2.0)


@pytest.fixture()
def mixture():
    from bayesflow.distributions import DiagonalNormal, DiagonalStudentT, Mixture

    return Mixture([DiagonalNormal(mean=1.0, std=2.0), DiagonalStudentT(df=25, mean=1.0, std=2.0)])


@pytest.fixture(params=["diagonal_normal", "diagonal_student_t", "mixture"])
def distribution(request):
    name, kwargs = request.param

    match name:
        case "diagonal_normal":
            from bayesflow.distributions import DiagonalNormal

            return DiagonalNormal(mean=1.0, std=2.0, **kwargs)
        case "diagonal_student_t":
            from bayesflow.distributions import DiagonalStudentT

            return DiagonalStudentT(df=10, loc=1.0, scale=2.0, **kwargs)
        case "mixture":
            from bayesflow.distributions import DiagonalNormal, DiagonalStudentT, Mixture

            return Mixture(
                [
                    DiagonalNormal(mean=1.0, std=2.0, trainable_parameters=True),
                    DiagonalStudentT(df=25, mean=1.0, std=2.0),
                ],
                **kwargs,
            )
    return request.getfixturevalue(request.param)
