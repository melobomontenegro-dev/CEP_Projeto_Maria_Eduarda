# Avaliação CEP + ML — classificação de variedades de grãos de feijão

Trabalho individual da disciplina de Controle Estatístico de Processos do Departamento de Engenharia de Produção da Universidade de Brasília, ministrada pelo Prof. Andre Luiz Marques Serrano.

Autora: Maria Eduarda Lobo Montenegro (matrícula 200033972). Abril de 2026.

## O que tem aqui

Este repositório contém o material da avaliação individual da disciplina. O enunciado pedia que cada aluno aplicasse a metodologia integrada de Controle Estatístico de Processos e aprendizado de máquina ao seu próprio conjunto de dados, diferente do que foi discutido em sala (Steel Plates Faults).

Escolhi o **Dry Bean Dataset** (Koklu & Ozkan, 2020), disponível no UCI Machine Learning Repository com o id 602. São 13.611 grãos de feijão fotografados por uma câmera industrial, descritos por 16 atributos morfológicos e classificados em 7 variedades comerciais. A motivação foi manter o paralelo metodológico com o modelo de Steel Plates (também multiclasse, também features de imagem, também controle de qualidade) mudando o domínio para algo mais próximo da agroindústria brasileira.

## Estrutura

```
CEP_Projeto_Maria_Eduarda/
├── avaliacao_qualidade_graos.ipynb   notebook completo com a execução final
├── testes_iniciais.ipynb             entrega parcial 1 (EDA básica)
├── RELATORIO_TECNICO_COMPLETO.md     relatório em markdown
├── RELATORIO_TECNICO_COMPLETO.pdf    versão diagramada do relatório
├── figuras/                          gráficos exportados do notebook
├── README.md                         este arquivo
└── requirements.txt                  dependências Python
```

## Como reproduzir

A maneira mais simples é abrir o notebook diretamente no Google Colab — toda a execução foi feita lá, e o dataset é carregado direto do UCI por meio do pacote `ucimlrepo`, sem dependência de arquivos locais.

Localmente, basta clonar o repositório, instalar as dependências de `requirements.txt` e abrir o `.ipynb` em qualquer Jupyter:

```
pip install -r requirements.txt
jupyter notebook avaliacao_qualidade_graos.ipynb
```

A semente aleatória está fixada em 42 em todos os componentes estocásticos, então os resultados devem ser idênticos.

## Resultados em uma linha

O modelo de melhor desempenho na validação cruzada foi o SVM com núcleo RBF (F1-macro de 0,9410). O Random Forest, mesmo após otimização via GridSearchCV, ficou em segundo lugar (0,9036 no teste). A análise de capacidade do processo indicou Cpk entre 0,38 e 0,47 — incapaz, segundo Montgomery (2020), mas o trabalho discute por que esse achado decorre mais da heterogeneidade entre variedades do que de instabilidade real do processo.

## Referência principal

Montgomery, D. C. (2020). *Introdução ao Controle Estatístico da Qualidade*, 8ª edição. Wiley. Capítulos 4 a 8 usados como base teórica.
