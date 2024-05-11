#Importações e configurações iniciais
import numpy as np
import pandas as pd
import google.generativeai as genai

GOOGLE_API_KEY=""
genai.configure(api_key=GOOGLE_API_KEY)

# for m in genai.list_models():
#   if 'embedContent' in m.supported_generation_methods:
#     print(m.name)

#Exemplo de embedding
# title = "A próxima geração de IA para desenvolvedores e Google Workspace"
# sample_text = ("Título: A próxima geração de IA para desenvolvedores e Google Workspace"
#     "\n"
#     "Artigo completo:\n"
#     "\n"
#     "Gemini API & Google AI Studio: Uma maneira acessível de explorar e criar protótipos com aplicações de IA generativa")

# embeddings = genai.embed_content(model="models/embedding-001",
#                                  content=sample_text,
#                                  title=title,
#                                  task_type="RETRIEVAL_DOCUMENT")

# print(embeddings)

#Listagem de documentos que serão buscados
DOCUMENT1 = {
    "Título": "Fazer chamada no NSA",
    "Conteúdo": "O NSA é o sistema de gestão escolar que permite fazer chamadas, lançar aulas e atividades e outras. Para fazer chamada use a Opção Chamada, escolha a Turma, o curso, confira a data e faça a chamada dos alunos marcando as faltas na tela da esquerda."}

DOCUMENT2 = {
    "Título": "Corrigir faltas",
    "Conteúdo": "É possível corrigir faltas sem a necessidade do coordenador, para isso: Acesse o Menu 'Área do professor e coordenador, em seguida Correções, Correções de faltas, escolha a data da aula e altere as faltas que foram lançadas erradas!'"}

# DOCUMENT3 = {
#     "Título": "Mudança de marchas",
#     "Conteúdo": "Seu Googlecar tem uma transmissão automática. Para trocar as marchas, basta mover a alavanca de câmbio para a posição desejada.  Park (Estacionar): Essa posição é usada quando você está estacionado. As rodas são travadas e o carro não pode se mover. Marcha à ré: Essa posição é usada para dar ré. Neutro: Essa posição é usada quando você está parado em um semáforo ou no trânsito. O carro não está em marcha e não se moverá a menos que você pressione o pedal do acelerador. Drive (Dirigir): Essa posição é usada para dirigir para frente. Low: essa posição é usada para dirigir na neve ou em outras condições escorregadias."}

documents = [DOCUMENT1,DOCUMENT2]

df = pd.DataFrame(documents)
df.columns = ["Titulo", "Conteudo"]
# print(df)
model = "models/embedding-001"
def embed_fn(title, text):
  return genai.embed_content(model=model,
                                 content=text,
                                 title=title,
                                 task_type="RETRIEVAL_DOCUMENT")["embedding"]
df["Embeddings"] = df.apply(lambda row: embed_fn(row["Titulo"], row["Conteudo"]), axis=1)
# print(df)

def gerar_e_buscar_consulta(consulta, base, model):
  embedding_da_consulta = genai.embed_content(model=model,
                                 content=consulta,
                                 task_type="RETRIEVAL_QUERY")["embedding"]

  produtos_escalares = np.dot(np.stack(df["Embeddings"]), embedding_da_consulta)

  indice = np.argmax(produtos_escalares)
  return df.iloc[indice]["Conteudo"]
consulta = "Lancei faltas erradas e agora ?"

trecho = gerar_e_buscar_consulta(consulta, df, model)
print(trecho)
generation_config = {
  "temperature": 0.5,
  "candidate_count": 1
}
prompt = f"Reescreva esse texto de uma forma mais descontraída, sem adicionar informações que não façam parte do texto: {trecho}"

model_2 = genai.GenerativeModel("gemini-1.0-pro",
                                generation_config=generation_config)
response = model_2.generate_content(prompt)
print(response.text)
