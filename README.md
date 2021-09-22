# tensorflow_log_loader

O Tensorflow é um dos frameworks de machine learning mais populares da atualidade. Ele utiliza o conceito de grafos para descrever o fluxo dos dados e operações do modelo. Cada nó representa uma operação matemática e cada conexão ou aresta do grafo representa uma array multidimensional, mais conhecida como tensor.

O Tensorboard é uma ferramenta que permite a visualização de qualquer estatística de uma rede neural como, por exemplo, parâmetros de treinamento (perda, acurácia e pesos), imagens e até o grafo construído. Isso pode ser muito útil para entender o fluxo dos tensores no grafo e assim debugar e otimizar o modelo.

Neste artigo, os log gerado pelo tensorborder de um modelo trainado, será carregado pelo python ao invés da plataforma tensorborder, e com isso podemos carregar dois ou mais logs gerando gráfico combinados

![N|Solid](https://github.com/henriqueburis/tensorflow_log_loader/blob/main/figure/ACC.png?raw=true)
