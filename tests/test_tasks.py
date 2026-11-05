"""Runner multiplataforma: cada subcomando se resuelve sin ejecutar nada pesado."""

import pytest

from fakenews import tasks


@pytest.mark.parametrize("name", ["setup", *tasks.TASKS])
def test_dry_run_of_every_task(name, capsys):
    assert tasks.run(name, dry_run=True) == 0
    assert capsys.readouterr().out


def test_required_tasks_exist():
    assert {"data", "train", "quickstart", "evaluate", "analyze", "test", "demo", "all"} <= set(tasks.TASKS)


def test_cli_commands_used_by_tasks_exist():
    from fakenews import cli

    source = open(cli.__file__, encoding="utf-8").read()
    for cmds in tasks.TASKS.values():
        for c in cmds:
            if "fakenews.cli" in c:
                assert f'"{c[3]}"' in source, c


def test_unknown_task_rejected():
    with pytest.raises(SystemExit):
        tasks.main(["inexistente"])
