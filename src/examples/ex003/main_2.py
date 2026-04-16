import operator
from dataclasses import dataclass
from typing import Annotated, Literal

import rich as r
from langgraph.graph import END, START, StateGraph


@dataclass
class State:
    nodes_path: Annotated[list[str], operator.add]
    current_number: int = 0


# 2. Definir Nós do Grafo
def node_a(state: State) -> State:
    print("> node_a", f"{state=}")
    return State(["A"], current_number=50)


def node_b(state: State) -> State:
    print("> node_b", f"{state=}")
    return State(["B"], current_number=state.current_number)


def node_c(state: State) -> State:
    print("> node_b", f"{state=}")
    return State(["C"], current_number=state.current_number)


# Função Condicional
def the_conditional(state: State) -> Literal["goes_to_b", "goes_to_c"]:
    if state.current_number >= 50:  # noqa: PLR2004
        return "goes_to_c"

    return "goes_to_b"


# 3. Instanciar Builder do Grafo
builder = StateGraph(State)

# 4. Adicionar nós do grafo
builder.add_node("A", node_a)
builder.add_node("B", node_b)
builder.add_node("C", node_c)

# 5. Adicionar arestas do grafo
builder.add_edge(START, "A")

# A edge pode ter um nome proprio, ou pode ser o nome dos nodes
builder.add_conditional_edges(
    "A", the_conditional, {"goes_to_b": "B", "goes_to_c": "C"}
)
builder.add_edge("B", END)
builder.add_edge("C", END)

# 6. Compilar o Grafo
graph = builder.compile()

# Exemplo PNG do Grafo
graph.get_graph().draw_mermaid_png(output_file_path="./src/examples/ex003/graph_02.png")

response = graph.invoke(State(nodes_path=[]))

print()
r.print(f"{response=}")
print()
