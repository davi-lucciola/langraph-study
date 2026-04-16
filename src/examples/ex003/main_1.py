import operator
from typing import Annotated, TypedDict

import rich as r
from langgraph.graph import StateGraph


# 1. Definir Estado do Grafo
class State(TypedDict):
    nodes_path: Annotated[list[str], operator.add]


# 2. Definir Nós do Grafo
def node_a(state: State) -> State:
    print("> node_a", f"{state=}")
    return {"nodes_path": ["A"]}


def node_b(state: State) -> State:
    print("> node_b", f"{state=}")
    return {"nodes_path": ["B"]}


# 3. Instanciar Builder do Grafo
builder = StateGraph(State)

# 4. Adicionar nós do grafo
builder.add_node("A", node_a)
builder.add_node("B", node_b)

# 5. Adicionar arestas do grafo
builder.add_edge("__start__", "A")
builder.add_edge("A", "B")
builder.add_edge("B", "__end__")

# 6. Compilar o Grafo
graph = builder.compile()

# Exemplo PNG do Grafo
# graph.get_graph().draw_mermaid_png(output_file_path="./src/ex003/graph_01.png")

response = graph.invoke({"nodes_path": []})

print()
r.print(f"{response=}")
print()
