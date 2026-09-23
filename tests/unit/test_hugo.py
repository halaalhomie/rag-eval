import pytest

from app.ingestion.hugo import GlossaryEntry, HugoShortcodeResolver, load_glossary

GLOSSARY = {
    "node": GlossaryEntry("node", "Node", "A node is a worker machine in Kubernetes.", "full"),
    "pod": GlossaryEntry(
        "pod",
        "Pod",
        'The smallest deployable unit, running on a {{< glossary_tooltip term_id="node" >}}.',
        "The smallest deployable unit.\n\nMore detail.",
    ),
}


@pytest.fixture
def resolve():
    return HugoShortcodeResolver("1.37", GLOSSARY)


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ('If a {{< glossary_tooltip term_id="node" >}} dies', "If a Node dies"),
        ('{{< glossary_tooltip text="Pods" term_id="pod" >}}', "Pods"),
        ('{{< glossary_tooltip term_id="missing" >}}', "missing"),
        ("v{{< skew currentVersion >}}", "v1.37"),
        ('{{< skew currentVersionAddMinor -1 "." >}}', "1.36"),
        ("{{< skew currentPatchVersion >}}", "1.37.x"),
        ('{{< param "version" >}}', "v1.37"),
        ('{{% heading "prerequisites" %}}', "Before you begin"),
        (
            '{{< feature-state for_k8s_version="v1.29" state="stable" >}}',
            "FEATURE STATE: Kubernetes v1.29 [stable]",
        ),
        (
            '{{< feature-state feature_gate_name="SidecarContainers" >}}',
            "FEATURE STATE: feature gate `SidecarContainers`",
        ),
        (
            '{{% code_sample file="pods/simple-pod.yaml" %}}',
            "(Example manifest: `pods/simple-pod.yaml`)",
        ),
        ('{{< include "task-tutorial-prereqs.md" >}}', ""),
        ("{{< version-check >}}", ""),
        ('{{< api-reference page="core/persistent-volume-v1" >}}', "PersistentVolume"),
        ('{{< link text="services" url="/docs/x/" >}}', "services"),
        ('[cfg]({{< relref "/docs/reference/kubelet" >}})', "[cfg](/docs/reference/kubelet)"),
    ],
)
def test_inline_shortcodes(resolve, src, expected):
    assert resolve(src) == expected


def test_admonition_becomes_label_and_keeps_content(resolve):
    out = resolve("{{< note >}}\nPods are ephemeral.\n{{< /note >}}")
    assert out.split() == ["Note:", "Pods", "are", "ephemeral."]


def test_glossary_definition_resolves_nested_shortcodes(resolve):
    out = resolve('{{< glossary_definition term_id="pod" prepend="A Pod is" >}}')
    assert out == "A Pod is the smallest deployable unit, running on a Node."


def test_glossary_definition_length_all(resolve):
    out = resolve('{{< glossary_definition term_id="pod" length="all" >}}')
    assert "More detail." in out


def test_comment_and_mermaid_blocks_are_dropped(resolve):
    src = "keep {{< comment >}}hidden{{< /comment >}} {{< mermaid >}}graph TD;A-->B{{< /mermaid >}}"
    assert resolve(src).split() == ["keep"]


def test_tabs_keep_names_and_content(resolve):
    out = resolve(
        '{{< tabs name="t" >}}{{% tab name="Linux" %}}apt install{{% /tab %}}{{< /tabs >}}'
    )
    assert out == "Linux:apt install"


def test_unknown_shortcodes_are_stripped_and_recorded(resolve):
    assert resolve("a {{< mystery x=1 >}}b{{< /mystery >}} c") == "a b c"
    assert resolve.unknown == {"mystery"}


def test_load_glossary_splits_short_and_full(tmp_path):
    (tmp_path / "_index.md").write_text("---\ntitle: Glossary\n---\n")
    (tmp_path / "node.md").write_text(
        "---\ntitle: Node\nid: node\n---\n A worker machine.\n\n<!--more-->\n\nMore.\n"
    )
    glossary = load_glossary(tmp_path)
    assert set(glossary) == {"node"}
    assert glossary["node"].short == "A worker machine."
    assert glossary["node"].full == "A worker machine.\n\nMore."
