# PROVA — Redes Neurais e Deep Learning (TI5A)

**Aluno(a):** GUSTAVO DUTRA TELLES  
**RA:** 245178 
**Professor:** André Insardi  
**Disciplina:** Deep Learning — Sistemas de Informação (TI5A)

---

## Questão 1 — Análise Conceitual e Arquitetural

### (a)Calcule o número total de parâmetros treináveis (pesos + bias) da camada de entrada para a primeira camada oculta na Abordagem A. Compare esse número com o de uma camada convolucional com 32 filtros 3×3 aplicada sobre a imagem original (3 canais). A partir dessa comparação, explique por que a CNN é mais eficiente para dados de imagem, mencionando os conceitos de compartilhamento de pesos e conectividade local. [Máx. 200 palavras + cálculos]

Cálculo MLP (entrada para 1ª camada oculta):
```text
N_in = 256 x 256 x 3 = 196.608
N_h1 = 512

Parametros = (N_in x N_h1) + N_h1
           = (196.608 x 512) + 512
           = 100.663.296 + 512
           = 100.663.808
```

Cálculo Conv2D (32 filtros 3x3, 3 canais):
```text
Parametros por filtro = (3 x 3 x 3) + 1
                      = 28

Total Conv2D = 28 x 32 = 896
```

Verificação em Python:
```python
N_in = 256 * 256 * 3
N_h1 = 512
params_mlp = N_in * N_h1 + N_h1

params_conv = (3 * 3 * 3 + 1) * 32

print("MLP:", params_mlp)      # 100663808
print("Conv2D:", params_conv)  # 896
```

Comparação: 100.663.808 vs 896 parâmetros.  
A CNN é mais eficiente por dois motivos: conectividade local (cada neurônio observa vizinhanças) e compartilhamento de pesos (um mesmo filtro percorre toda a imagem). Na MLP achatada, cada neurônio conecta com todos os pixels, elevando custo, risco de overfitting e perda de estrutura espacial. Como bordas e texturas se repetem em várias posições, a CNN reaproveita filtros e generaliza melhor com muito menos parâmetros.

### (b) Explique por que o Perceptron de camada única é incapaz de aprender a função XOR. Na sua resposta: (i) apresente os 4 pontos do XOR no plano cartesiano, (ii) demonstre geometricamente a impossibilidade de separação linear, e (iii) mostre como uma MLP com 1 camada oculta de 2 neurônios resolve o problema (descreva o papel de cada neurônio na transformação do espaço). [Máx. 250 palavras]

Pontos do XOR no plano cartesiano (x1, x2):

```text
x1  x2  XOR(x1,x2)
0   0   0
1   1   0
0   1   1
1   0   1
```

Leitura geométrica dos pontos:
- Classe 0: (0,0) e (1,1) (diagonal principal).
- Classe 1: (0,1) e (1,0) (diagonal secundária).

Geometricamente, as classes 1 ficam em cantos opostos do quadrado unitário, e as classes 0 nos outros dois cantos opostos. Não existe reta $w_1x_1+w_2x_2+b=0$ que satisfaça simultaneamente os quatro pontos do XOR; ao impor $f(0,0)=0$, $f(1,1)=0$, $f(1,0)=1$, $f(0,1)=1$, surgem desigualdades incompatíveis para um único hiperplano.

Com 2 neurônios ocultos, a MLP cria duas fronteiras lineares no espaço original (ex.: $x_1+x_2=0,5$ e $x_1+x_2=1,5$). Um neurônio pode codificar OR e outro AND; na saída, XOR = OR - AND (ou OR e não-AND). Assim, o mapeamento oculto $\phi(x)$ transforma o problema em separável linearmente no espaço $\phi(x)$.

### (c)No contexto deste problema de classificação de pragas, o dataset é desbalanceado. Discuta: (i) por que usar apenas acurácia como métrica pode ser enganoso, (ii) quais métricas alternativas seriam mais adequadas e por quê, e (iii) cite pelo menos duas técnicas de pré-processamento ou treinamento que poderiam mitigar o desbalanceamento. [Máx. 150 palavras]

Com classes desbalanceadas, acurácia isolada pode ser enganosa: o modelo pode acertar muito a classe majoritária e falhar nas raras. Métricas mais adequadas: macro-F1 (equilibra desempenho entre classes), recall por classe (detecta falhas nas classes raras), matriz de confusão e balanced accuracy.

Técnicas para mitigar desbalanceamento:  
1. Reamostragem (oversampling/augmentação das classes minoritárias e/ou undersampling da majoritária).  
2. Loss ponderada por classe ou focal loss.  
3. Mini-batches balanceados (amostragem estratificada).

---

## Questão 2 — Backpropagation: Compreensão e Diagnóstico

### (a) Para CADA cenário (A, B e C), faça um diagnóstico técnico completo: [Máx. 100 palavras por cenário, total 300 palavras]

(i)	Identifique o problema principal e explique a causa raiz relacionando com o algoritmo de backpropagation e/ou o processo de treinamento.
(ii)	Explique, usando conceitos do backpropagation (regra da cadeia, gradientes, atualização de pesos), POR QUE esse comportamento ocorre.
(iii)	Proponha uma solução específica e justifique por que ela resolveria o problema.


**Cenário A**  
Problema: loss estabiliza alta e acurácia baixa.  
Causa raiz: inicialização N(0,1) com sigmoid gera ativações saturadas; como sigmoid'(z) fica perto de 0 fora de z perto de 0, gradientes desaparecem no backward. Pela regra da cadeia, multiplicações de derivadas pequenas reduzem fortemente dL/dW.  
Solução: inicialização Xavier/Glorot, normalização de entrada e/ou ReLU nas ocultas para melhorar fluxo de gradiente.

**Cenário B**  
Problema: loss oscila e não converge.  
Causa raiz: learning rate 5.0 é excessivo; a atualização W <- W - eta * grad(L) dá passos grandes e ultrapassa mínimos.  
No backprop, gradientes corretos viram atualizações instáveis por eta alto.  
Solução: reduzir eta (ex.: 1e-3 a 1e-2), usar scheduler e Adam; opcionalmente gradient clipping para conter explosões.

**Cenário C**  
Problema: overfitting (treino 99,8% vs teste 72%).  
Causa raiz: alta capacidade da rede, poucos dados (1.000 imagens), 500 épocas e ausência de regularização.  
Backprop otimiza bem treino, mas memoriza padrões específicos e perde generalização.  
Solução: dropout, regularização L2, data augmentation, early stopping, redução de capacidade e/ou mais dados.

### (b) Considere a rede do colega (4 camadas ocultas, todas com sigmoid). Responda: [Máx. 250 palavras]

(i)	Desenhe o grafo computacional simplificado desta rede (pode ser em texto, ex: x → W1 → σ
→ W2 → ...) e explique como a regra da cadeia é aplicada para calcular ∂L/∂W1 (pesos da
primeira camada). Quantas multiplicações encadeadas de derivadas são necessárias?
(ii)	Sabendo que a derivada da sigmoid tem valor máximo de 0.25 (quando z=0), explique o que acontece quando multiplicamos 4 dessas derivadas em sequência durante o backward pass. Como isso se relaciona com o problema do Cenário A?
(iii)	Proponha uma alteração na arquitetura (função de ativação e/ou técnica) que mitigaria esse problema. Justifique matematicamente por que sua proposta é superior.


(i) Grafo simplificado:  
x -> W1 -> sigma -> W2 -> sigma -> W3 -> sigma -> W4 -> sigma -> Wo -> softmax -> L.

Para $\partial L/\partial W_1$, a regra da cadeia escreve:
$$
\frac{\partial L}{\partial W_1}=\frac{\partial L}{\partial a_4}\frac{\partial a_4}{\partial z_4}\frac{\partial z_4}{\partial a_3}\cdots\frac{\partial a_1}{\partial z_1}\frac{\partial z_1}{\partial W_1}
$$
Há 4 fatores explícitos de $\sigma'(z_l)$ das camadas ocultas, além dos termos lineares das matrizes de pesos.

(ii) Cálculo do encolhimento do gradiente:
```text
max(sigmoid'(z)) = 0,25
Produto em 4 camadas sigmoid = 0,25^4
                            = 0,00390625
```
Verificação em Python:
```python
produto = 0.25 ** 4
print(produto)  # 0.00390625
```
Logo, mesmo no melhor caso, o gradiente já é pequeno; em saturação, fica menor ainda. Isso explica o comportamento do Cenário A (vanishing gradients).

(iii) Alteração: trocar sigmoid por ReLU/LeakyReLU nas ocultas + inicialização He (e, se possível, BatchNorm). ReLU tem derivada 1 na região ativa, preservando melhor magnitude de gradiente; He ajuda a manter variância estável no forward/backward.

### (c) O colega quer entender a diferença entre calcular gradientes por "enumeração de caminhos" no grafo computacional vs. o algoritmo de backpropagation. Explique: [Máx. 200 palavras]

(i)	O que é o Pathwise Aggregation Lemma e como ele fundamenta o backpropagation.
(ii)	Por que a enumeração direta de caminhos tem complexidade exponencial O(2^n), enquanto o backprop é linear O(n).
(iii)	Dê um exemplo concreto com a rede de 4 camadas: quantos caminhos existiriam entre a perda L e um peso da primeira camada? Como o backprop evita essa explosão?


(i) O Pathwise Aggregation Lemma diz que
$$
\frac{\partial L}{\partial u}=\sum_{p\in\mathcal{P}(u\to L)}\prod_{(i\to j)\in p}\frac{\partial z_j}{\partial z_i}
$$
ou seja: a derivada total é a soma das contribuições de todos os caminhos, e cada contribuição é um produto de derivadas locais. Isso é a base matemática do backprop.

(ii) Enumerar todos os caminhos cresce combinatoriamente com profundidade e largura (na prática, inviável). O backprop evita isso ao reutilizar subresultados: calcula um único $\delta_l=\partial L/\partial z_l$ por neurônio/camada e propaga recursivamente ($\delta_l=W_{l+1}^T\delta_{l+1}\odot\sigma'(z_l)$). Assim, o custo fica proporcional ao número de arestas/operações do grafo (linear no tamanho da rede para um forward+backward).

(iii) Na rede 784->256->128->64->10, para um peso em W1:
```text
Numero de caminhos ate L = 128 x 64 x 10 = 81.920
```
Verificação em Python:
```python
caminhos = 128 * 64 * 10
print(caminhos)  # 81920
```
O backprop evita essa explosão ao propagar erros camada a camada, sem enumerar caminhos individualmente.

---

## Questão 3 — Depuração de Código: Backpropagation

### (a) Identifique e corrija os 5 erros presentes no código (linhas A, B, C, D e E estão marcadas como dicas de localização — nem todas as linhas marcadas contêm erro, e pode haver erros em linhas não marcadas). Para cada erro encontrado: [Máx. 60 palavras por erro, total 300 palavras + código]

(i)	Indique a linha e transcreva o trecho com erro.
(ii)	Explique conceitualmente por que está errado e qual o impacto no treinamento.
(iii)	Apresente o código corrigido.


**Erro 1 (linha 11)**  
Trecho: `def sigmoid_deriv(z): return sigmoid(z)`  
Problema: derivada incorreta.  
Correção:
```python
def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)
```

**Erro 2 (linhas 13–14)**  
Trecho: `exp_z = np.exp(z); return exp_z / np.sum(exp_z)`  
Problema: instabilidade numérica (overflow).  
Correção:
```python
def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)
```

**Erro 3 (linhas 19–20)**  
Trecho: pesos com `np.random.randn(...)` sem escala.  
Problema: ativações sigmoid saturam no início.  
Correção:
```python
W1 = np.random.randn(4, 5) * 0.01
W2 = np.random.randn(5, 3) * 0.01
```

**Erro 4 (linha 39)**  
Trecho: `delta2 = y_true - a2`  
Problema: sinal inconsistente para gradiente de softmax + cross-entropy.  
Correção:
```python
delta2 = a2 - y_true
```

**Erro 5 (linhas 45–46)**  
Trecho: `W += alpha * grad`  
Problema: atualização em direção de subida.  
Correção:
```python
W2 -= alpha * grad_W2
W1 -= alpha * grad_W1
```

Código consolidado corrigido:
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
X = MinMaxScaler().fit_transform(iris.data)
y = iris.target
one_hot = np.eye(3)[y]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-15))

np.random.seed(42)
W1 = np.random.randn(4, 5) * 0.01
W2 = np.random.randn(5, 3) * 0.01

alpha = 0.01

for epoch in range(500):
    total_loss = 0.0
    for i in range(len(X)):
        x = X[i]
        y_true = one_hot[i]

        z1 = np.dot(x, W1)
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2)
        a2 = softmax(z2)

        total_loss += cross_entropy(y_true, a2)

        delta2 = a2 - y_true
        delta1 = np.dot(delta2, W2.T) * sigmoid_deriv(z1)

        grad_W2 = np.outer(a1, delta2)
        grad_W1 = np.outer(x, delta1)

        W2 -= alpha * grad_W2
        W1 -= alpha * grad_W1

    print(f"Epoch {epoch} | Loss: {total_loss:.4f}")
```

### (b) Após corrigir todos os erros, o colega questiona: "Por que usamos softmax na saída e não sigmoid, se sigmoid também dá valores entre 0 e 1?". Responda de forma rigorosa, explicando a diferença matemática e prática entre as duas funções quando aplicadas à camada de saída em classificação multiclasse. Inclua na resposta por que a combinação softmax + cross-entropy simplifica o cálculo do gradiente na camada de saída. [Máx. 150 palavras]

Em multiclasse exclusiva, a saída deve modelar uma variável categórica. A softmax define
$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}},\quad \sum_k p_k=1
$$
logo os logits são acoplados: aumentar $z_k$ altera todas as probabilidades. Já a sigmoid modela Bernoullis independentes por classe,
$$
p_k=\sigma(z_k)
$$
sem normalização conjunta; isso é adequado para multirrótulo, não para classe única mutuamente exclusiva.

Com softmax + cross-entropy (likelihood categórica), o gradiente por logit simplifica para:
```text
dL/dz_k = p_k - y_k
```
Isso torna o backward direto, estável e estatisticamente coerente com decisão competitiva entre classes.

---

## Questão 4 — Redes Neurais Convolucionais (CNN)

### (a) ] Calcule as dimensões espaciais (altura × largura × canais) da saída de cada camada, desde a entrada (28×28×1) até o Flatten (camada 7). Mostre a fórmula utilizada para o cálculo da dimensão após convolução e após pooling. Qual é o tamanho do vetor após o Flatten? [Máx. 100 palavras + tabela/cálculos]

Fórmula:
```text
dim_out = floor((dim_in - k + 2p) / s) + 1
```

Tabela de dimensões:
```text
Entrada                             : 28 x 28 x 1
Conv1 (3x3, s=1, p=0, 32 filtros)   : 26 x 26 x 32
ReLU                                : 26 x 26 x 32
MaxPool (2x2, s=2)                  : 13 x 13 x 32
Conv2 (3x3, s=1, p=0, 64 filtros)   : 11 x 11 x 64
ReLU                                : 11 x 11 x 64
MaxPool (2x2, s=2)                  :  5 x  5 x 64
Flatten                             : 5 x 5 x 64 = 1600
```

Verificação em Python:
```python
import math

def out_dim(n, k, s, p=0):
    return math.floor((n - k + 2 * p) / s) + 1

h = w = 28
h = w = out_dim(h, 3, 1, 0)   # Conv1
h = w = out_dim(h, 2, 2, 0)   # Pool1
h = w = out_dim(h, 3, 1, 0)   # Conv2
h = w = out_dim(h, 2, 2, 0)   # Pool2
flatten = h * w * 64
print((h, w, 64), flatten)    # (5, 5, 64) 1600
```

### (b)  Calcule o número total de parâmetros treináveis (pesos + bias) em cada camada convolucional e em cada camada densa. Apresente o total geral da rede. [Máx. 80 palavras + cálculos]

Para Conv2D: params = (kernel_h × kernel_w × canais_entrada + 1) × num_filtros

```text
Conv1    = (3 x 3 x 1 + 1) x 32   = (9 + 1) x 32      = 320
Conv2    = (3 x 3 x 32 + 1) x 64  = (288 + 1) x 64    = 18.496
Dense128 = (1600 x 128) + 128                           = 204.928
Dense10  = (128 x 10) + 10                              = 1.290

Total = 320 + 18.496 + 204.928 + 1.290 = 225.034 parametros
```

Verificação em Python:
```python
conv1 = (3 * 3 * 1 + 1) * 32
conv2 = (3 * 3 * 32 + 1) * 64
dense128 = 1600 * 128 + 128
dense10 = 128 * 10 + 10
total = conv1 + conv2 + dense128 + dense10
print(conv1, conv2, dense128, dense10, total)
# 320 18496 204928 1290 225034
```

### (c) Um colega sugere remover todas as camadas de pooling e compensar aumentando o stride das convoluções para 2. Analise criticamente esta proposta: [Máx. 200 palavras + cálculos]

(i)	Qual seria a nova dimensão espacial após cada convolução?
(ii)	Compare o número de parâmetros treináveis com a arquitetura original.
(iii)	Discuta os trade-offs em termos de invariância a translação, perda de informação espacial e custo computacional.


Sem pooling e com stride 2 nas convoluções:

(i) Dimensões espaciais  
```text
Conv1: floor((28 - 3) / 2) + 1 = 13  -> 13 x 13 x 32
Conv2: floor((13 - 3) / 2) + 1 =  6  ->  6 x  6 x 64
Flatten: 6 x 6 x 64 = 2304
```

(ii) Parâmetros  
```text
Convs: 320 e 18.496 (iguais)
Dense128: (2304 x 128) + 128 = 295.040
Dense10: 1.290
Total novo = 320 + 18.496 + 295.040 + 1.290 = 315.146
Aumento = 315.146 - 225.034 = 90.112
```

Verificação em Python:
```python
conv1 = 320
conv2 = 18496
dense128_novo = 2304 * 128 + 128
dense10 = 1290
total_novo = conv1 + conv2 + dense128_novo + dense10

total_original = 225034
aumento = total_novo - total_original

print(total_novo, aumento)
# 315146 90112
```

(iii) Trade-offs  
Stride 2 reduz resolução mais agressivamente e pode perder detalhes finos. Pooling tende a oferecer invariância local de translação de forma explícita. Embora stride 2 possa reduzir custo de algumas ativações, aqui o flatten cresce (2304 vs 1600), aumentando parâmetros da cabeça densa e risco de overfitting. Portanto, a troca não é automaticamente melhor; depende de desempenho empírico e regularização.

---

## Questão 5 — Questão Integradora (Ensaio)

### Você foi contratado(a) como consultor(a) técnico(a) para avaliar a viabilidade de um projeto de IA em uma empresa de seguros. O projeto visa classificar automaticamente sinistros de veículos a partir de fotografias dos danos enviadas pelos segurados via aplicativo móvel. O modelo deve categorizar o dano em 4 níveis de severidade.

### Redija um parecer técnico que aborde obrigatoriamente todos os seguintes pontos. [Máximo 500 palavras no total — seja direto e técnico]

### 1.	Pipeline de dados: Descreva as etapas de pré-processamento necessárias para as imagens (normalização, redimensionamento, data augmentation). Justifique cada escolha.

### 2.	Arquitetura do modelo: Proponha uma arquitetura de rede neural adequada ao problema. Justifique a escolha entre MLP, CNN ou outra abordagem. Detalhe as camadas, funções de ativação e a função de perda.

### 3.	Treinamento: Explique o papel do backpropagation no treinamento desta rede. Descreva como o gradiente é propagado de trás para frente usando a regra da cadeia, e por que a programação dinâmica torna isso computável. Discuta a escolha da taxa de aprendizado e potenciais problemas (vanishing/exploding gradients) e como mitigá-los.

### 4.	Ética e privacidade: Considerando que as fotos podem conter placas de veículos, rostos de pessoas e localizações, discuta os riscos éticos e de privacidade (LGPD). Proponha medidas de mitigação.
        
        ### Critérios de avaliação: profundidade técnica (5 pts), integração entre os tópicos (4 pts), coerência e redação (3 pts), originalidade da análise (3 pts). Respostas genéricas ou copiadas de IA sem personalização receberão até 50% da pontuação.



O projeto é viável, mas não é apenas um problema de acurácia: é um problema de decisão de risco com impacto financeiro e regulatório.

No pipeline, as imagens devem ser redimensionadas (ex.: 224x224), normalizadas ([0,1] ou estatísticas do backbone) e submetidas a augmentations realistas (brilho/contraste, rotação leve, zoom e compressão), refletindo a captura via app. O split precisa ser estratificado por severidade e sem vazamento (imagens do mesmo sinistro não podem cair em treino e teste).

A melhor escolha é uma CNN com transfer learning (ResNet18/EfficientNet-B0), superior a MLP para imagem por preservar estrutura espacial. Arquitetura proposta: backbone convolucional + GlobalAveragePooling + Dense(128, ReLU) + Dropout(0,3) + Dense(4, Softmax). Perda: cross-entropy ponderada por classe (ou focal loss em desbalanceamento severo). Eu também usaria matriz de custo: subestimar dano grave deve custar mais do que superestimar dano leve.

No treino, o backprop propaga o erro da saída para as camadas anteriores pela regra da cadeia. O algoritmo usa programação dinâmica (deltas por camada), evitando enumerar caminhos. Para estabilidade e generalização: AdamW com scheduler, ReLU, inicialização He, BatchNorm, gradient clipping, early stopping, weight decay e validação estratificada.

Em produção, não basta acurácia: monitorar macro-F1 por classe, matriz de confusão, calibração (ECE/Brier) e drift de dados (PSI/KL em brilho, resolução e distribuição de classes). Se confiança < limiar, encaminhar para revisão humana. Recalibração periódica (temperature scaling) e retreino com janela móvel reduzem degradação. Ponto crítico: a qualidade do rótulo (severidade) tende a variar entre peritos; sem governança de anotação, o teto do modelo fica artificialmente baixo.

Em ética e LGPD, os principais riscos são exposição de placas, rostos e metadados, além de viés por região/tipo de veículo. Mitigações: blur automático de placas/rostos, remoção de EXIF sensível, criptografia em trânsito/repouso, controle de acesso por perfil, retenção mínima, trilha de auditoria, base legal explícita e canal de contestação com revisão humana. Na prática, eu trataria o sistema como suporte ao analista (não árbitro final) até evidência robusta de desempenho estável por segmento.
