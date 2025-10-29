# utils/analysis.py
import re
import json
from collections import Counter
from wordcloud import WordCloud
import networkx as nx
from openai import OpenAI
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

_PT_STOPWORDS = {

    "a", "à", "às", "ao", "aos", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "dá", "dão", "em", "no", "na", "nos", "nas",
    "por", "para", "pra", "com", "sem", "entre", "sobre", "sob", "até", "após",
    "que", "se", "é", "ser", "são", "era", "eram", "foi", "fui", "vai", "vou", "e",
    "ou", "mas", "como", "quando", "onde", "qual", "quais", "porque", "porquê",
    "já", "não", "sim", "também", "mais", "menos", "muito", "muita", "muitos",
    "muitas", "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas",
    "depois", "antes", "este", "esta", "estes", "estas", "isso", "isto", "aquele",
    "aquela", "aqueles", "aquelas", "lhe", "lhes", "ele", "ela", "eles", "elas",
    "você", "vocês", "nós", "nosso", "nossa", "nossos", "nossas", 'adeus', 'agora',
    'aí', 'ainda', 'além', 'algo', 'alguém', 'algum', 'alguma', 'algumas', 'alguns',
    'ali', 'ampla', 'amplas', 'amplo', 'amplos', 'ano', 'anos', 'ante', 'apenas',
    'apoio', 'aqui', 'aquilo', 'área', 'assim', 'atrás', 'através', 'baixo', 'bastante',
    'bem', 'boa', 'boas', 'bom', 'bons', 'breve', 'cá', 'cada', 'catorze', 'cedo',
    'cento', 'certamente', 'certeza', 'cima', 'cinco', 'coisa', 'coisas', 'conselho',
    'contra', 'contudo', 'custa', 'debaixo', 'dela', 'delas', 'dele', 'deles',
    'demais', 'dentro', 'desde', 'dessa', 'dessas', 'desse', 'desses', 'desta',
    'destas', 'deste', 'destes', 'deve', 'devem', 'devendo', 'dever', 'deverá',
    'deverão', 'deveria', 'deveriam', 'devia', 'deviam', 'dez', 'dezanove',
    'dezasseis', 'dezassete', 'dezoito', 'dia', 'diante', 'disse', 'disso',
    'disto', 'dito', 'diz', 'dizem', 'dizer', 'dois', 'doze', 'duas', 'dúvida',
    'embora', 'enquanto', 'éramos', 'és', 'essa', 'essas', 'esse', 'esses', 'está',
    'estamos', 'estão', 'estar', 'estás', 'estava', 'estavam', 'estávamos', 'esteja',
    'estejam', 'estejamos', 'esteve', 'estive', 'estivemos', 'estiver', 'estivera',
    'estiveram', 'estivéramos', 'estiverem', 'estivermos', 'estivesse', 'estivessem',
    'estivéssemos', 'estiveste', 'estivestes', 'estou', 'etc', 'eu', 'exemplo',
    'faço', 'falta', 'favor', 'faz', 'fazeis', 'fazem', 'fazemos', 'fazendo', 'fazer',
    'fazes', 'feita', 'feitas', 'feito', 'feitos', 'fez', 'fim', 'final', 'fomos',
    'for', 'fora', 'foram', 'fôramos', 'forem', 'forma', 'formos', 'fosse', 'fossem',
    'fôssemos', 'foste', 'fostes', 'geral', 'grande', 'grandes', 'grupo', 'há',
    'haja', 'hajam', 'hajamos', 'hão', 'havemos', 'havia', 'hei', 'hoje', 'hora',
    'horas', 'houve', 'houvemos', 'houver', 'houvera', 'houverá', 'houveram',
    'houvéramos', 'houverão', 'houverei', 'houverem', 'houveremos', 'houveria',
    'houveriam', 'houveríamos', 'houvermos', 'houvesse', 'houvessem', 'houvéssemos',
    'la', 'lá', 'lado', 'lo', 'local', 'logo', 'longe', 'lugar', 'maior', 'maioria',
    'mal', 'máximo', 'me', 'meio', 'menor', 'mês', 'meses', 'mesma', 'mesmas',
    'mesmo', 'mesmos', 'nada', 'naquela', 'naquelas', 'naquele', 'naqueles', 'nem',
    'nenhum', 'nenhuma', 'nessa', 'nessas', 'nesse', 'nesses', 'nesta', 'nestas',
    'neste', 'nestes', 'ninguém', 'nível', 'noite', 'nome', 'nova', 'novas', 'nove',
    'novo', 'novos', 'num', 'numa', 'número', 'nunca', 'obra', 'obrigada', 'obrigado',
    'oitava', 'oitavo', 'oito', 'ontem', 'onze', 'outra', 'outras', 'outro', 'outros',
    'parece', 'parte', 'partir', 'paucas', 'pela', 'pelas', 'pelo', 'pelos',
    'pequena', 'pequenas', 'pequeno', 'pequenos', 'per', 'perante', 'perto',
    'pode', 'pude', 'pôde', 'podem', 'podendo', 'poder', 'poderia', 'poderiam',
    'podia', 'podiam', 'põe', 'põem', 'pois', 'ponto', 'pontos', 'porém', 'posição',
    'possível', 'possivelmente', 'posso', 'pouca', 'poucas', 'pouco', 'poucos',
    'primeira', 'primeiras', 'primeiro', 'primeiros', 'própria', 'próprias',
    'próprio', 'próprios', 'próxima', 'próximas', 'próximo', 'próximos', 'puderam',
    'quáis', 'quanto', 'quantos', 'quarta', 'quarto', 'quatro', 'quê', 'quem',
    'quer', 'quereis', 'querem', 'queremas', 'queres', 'quero', 'questão', 'quinta',
    'quinto', 'quinze', 'relação', 'sabe', 'sabem', 'segunda', 'segundo', 'sei',
    'seis', 'seja', 'sejam', 'sejamos', 'sempre', 'sendo', 'será', 'serão',
    'serei', 'seremos', 'seria', 'seriam', 'seríamos', 'sete', 'sétima', 'sétimo',
    'sexta', 'sexto', 'si', 'sido', 'sistema', 'só', 'sois', 'somos', 'sou',
    'tal', 'talvez', 'tampouco', 'tanta', 'tantas', 'tanto', 'tão', 'tarde',
    'te', 'tem', 'tém', 'têm', 'temos', 'tendes', 'tendo', 'tenha', 'tenham',
    'tenhamos', 'tenho', 'tens', 'ter', 'terá', 'terão', 'terceira', 'terceiro',
    'terei', 'teremos', 'teria', 'teriam', 'teríamos', 'teu', 'teus', 'teve',
    'ti', 'tido', 'tinha', 'tinham', 'tínhamos', 'tive', 'tivemos', 'tiver',
    'tivera', 'tiveram', 'tivéramos', 'tiverem', 'tivermos', 'tivesse',
    'tivessem', 'tivéssemos', 'tiveste', 'tivestes', 'toda', 'todas', 'todavia',
    'todo', 'todos', 'trabalho', 'três', 'treze', 'tu', 'tua', 'tuas', 'tudo',
    'última', 'últimas', 'último', 'últimos', 'vais', 'vão', 'vários', 'vem',
    'vêm', 'vendo', 'vens', 'ver', 'vez', 'vezes', 'viagem', 'vindo', 'vinte',
    'vir', 'vos', 'vós', 'vossa', 'vossas', 'vosso', 'vossos', 'zero',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '_'

}

def tokenize_pt(texto: str):
    texto = texto.lower()
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]+", texto)
    return [t for t in tokens if len(t) >= 3 and t not in _PT_STOPWORDS]

def gerar_wordcloud(corpus_text: str):
    wc = WordCloud(width=450, height=280, background_color="white", collocations=False)
    wc.generate(corpus_text)
    img = wc.to_image()
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def build_word_graph(token_sequences, min_edge_weight: int = 1):
    G = nx.Graph()
    node_counts = Counter()
    edge_counts = Counter()
    for seq in token_sequences:
        node_counts.update(seq)
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a != b:
                edge = tuple(sorted((a, b)))
                edge_counts[edge] += 1
    for w, c in node_counts.items():
        G.add_node(w, count=int(c))
    for (a, b), w in edge_counts.items():
        if w >= min_edge_weight:
            G.add_edge(a, b, weight=int(w))
    return G

def score_from_label(label: str, confidence: float) -> float:
    sgn = 1 if label == "positivo" else (-1 if label == "negativo" else 0)
    try:
        c = float(confidence)
    except Exception:
        c = 0.0
    return round(sgn * max(0.0, min(1.0, c)), 3)

def _formatar_prompt_sentimento(texto: str) -> str:
    return (
        "Você é um classificador de sentimento. Classifique a mensagem a seguir.\n"
        "Responda APENAS com JSON válido com as chaves exatamente assim:\n"
        '{"label":"positivo|neutro|negativo","confidence":0.0-1.0,"emotions":["..."],"reason":"..."}\n'
        "Mensagem:\n"
        f"{texto.strip()}"
    )

def analisar_sentimento(texto: str, modelo_sentimento: str = "gpt-4.1-mini"):
    try:
        resp = client.chat.completions.create(
            model=modelo_sentimento,
            messages=[
                {"role": "system", "content": "Retorne JSON estrito."},
                {"role": "user", "content": _formatar_prompt_sentimento(texto)}
            ],
            temperature=0.0,
            max_tokens=150,
            top_p=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        label = str(data.get("label", "neutro")).lower()
        conf = float(data.get("confidence", 0.5))
        emotions = data.get("emotions", [])
        reason = str(data.get("reason", ""))
        return {
            "label": label if label in {"positivo", "neutro", "negativo"} else "neutro",
            "confidence": max(0.0, min(1.0, conf)),
            "emotions": emotions if isinstance(emotions, list) else [str(emotions)],
            "reason": reason.strip()
        }
    except Exception as e:
        return {
            "label": "neutro",
            "confidence": 0.0,
            "emotions": [],
            "reason": f"Erro: {str(e)}"
        }

def processar_lista_mensagens(mensagens: list[str]):
    corpus_tokens = []
    texto_completo = ""
    sentimentos = []
    for msg in mensagens:
        tokens = tokenize_pt(msg)
        if tokens:
            corpus_tokens.append(tokens)
            texto_completo += " " + " ".join(tokens)
        sent = analisar_sentimento(msg)
        sent["score"] = score_from_label(sent["label"], sent["confidence"])
        sentimentos.append(sent)
    grafo = build_word_graph(corpus_tokens)
    wordcloud_img = gerar_wordcloud(texto_completo)
    return sentimentos, grafo, wordcloud_img
