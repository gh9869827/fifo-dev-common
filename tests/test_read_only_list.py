import pytest
from fifo_dev_common.containers.read_only.read_only_list import ReadOnlyList


@pytest.fixture(name="sample_data")
def sample_data_impl() -> list[str]:
    return ['a', 'w', 's', 'o', 'm', 'e']


@pytest.fixture(name="readonly")
def readonly_impl(sample_data: list[str]) -> ReadOnlyList[str]:
    return ReadOnlyList(sample_data)


def test_len(readonly: ReadOnlyList[str]) -> None:
    assert len(readonly) == 6


def test_get_item(readonly: ReadOnlyList[str]) -> None:
    assert readonly[0] == 'a'
    assert readonly[-1] == 'e'


def test_get_slice(readonly: ReadOnlyList[str]) -> None:
    assert readonly[1:4] == ['w', 's', 'o']


def test_iter(readonly: ReadOnlyList[str]) -> None:
    assert list(iter(readonly)) == ['a', 'w', 's', 'o', 'm', 'e']

def test_index_with_stop(readonly: ReadOnlyList[str]) -> None:
    assert readonly.index('s', 0, 4) == 2

    # 'm' appears at index 4 so this should fail because it's outside stop=4
    with pytest.raises(ValueError):
        readonly.index('m', 0, 4)

def test_count(readonly: ReadOnlyList[str]) -> None:
    assert readonly.count('o') == 1
    assert readonly.count('z') == 0


def test_index(readonly: ReadOnlyList[str]) -> None:
    assert readonly.index('w') == 1
    assert readonly.index('o', 2) == 3
    with pytest.raises(ValueError):
        readonly.index('z')
    with pytest.raises(ValueError):
        readonly.index('o', 4)


def test_repr(readonly: ReadOnlyList[str]) -> None:
    assert repr(readonly) == "ReadOnlyList(['a', 'w', 's', 'o', 'm', 'e'])"


def test_eq_with_same_data(sample_data: list[str]) -> None:
    a = ReadOnlyList(sample_data)
    b = ReadOnlyList(sample_data.copy())
    assert a == b


def test_eq_with_sequence(sample_data: list[str]) -> None:
    ro = ReadOnlyList(sample_data)
    assert ro == sample_data


def test_eq_with_different_type(readonly: ReadOnlyList[str]) -> None:
    assert readonly != 123
    assert readonly is not None


def test_contains(readonly: ReadOnlyList[str]) -> None:
    assert 's' in readonly
    assert 'x' not in readonly


def test_no_mutation_allowed(readonly: ReadOnlyList[str]) -> None:
    with pytest.raises(AttributeError):
        readonly.append('!')  # type: ignore[attr-defined]

    with pytest.raises(TypeError):
        readonly[0] = 'Z'  # type: ignore[index]

    with pytest.raises(TypeError):
        del readonly[1]  # type: ignore[attr-defined]
