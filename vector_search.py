import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
import os
import pickle
from typing import List, Optional

class VectorSearch:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 index_path: str = 'faiss_index.bin', 
                 docs_path: str = 'documents.pkl'):
        """
        Инициализация векторного поиска с использованием Sentence Transformers и FAISS.
        
        Args:
            model_name: Название модели Sentence Transformers
            index_path: Путь для сохранения/загрузки индекса FAISS
            docs_path: Путь для сохранения/загрузки документов
        """
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.docs_path = docs_path
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        
        # Пытаемся загрузить существующий индекс и документы
        self._load_index()
    
    def _load_index(self):
        """Загрузка индекса и документов с диска, если они существуют"""
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            print("Загрузка индекса FAISS и документов...")
            self.index = faiss.read_index(self.index_path)
            with open(self.docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Загружено {len(self.documents)} документов")
        else:
            # Создаем новый пустой индекс
            self.index = faiss.IndexFlatL2(self.dimension)
            print("Создан новый индекс FAISS")
    
    def save_index(self):
        """Сохранение индекса и документов на диск"""
        if self.index is not None:
            print("Сохранение индекса FAISS...")
            faiss.write_index(self.index, self.index_path)
            with open(self.docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            print(f"Сохранено {len(self.documents)} документов")
    
    def add_documents(self, texts: List[str], batch_size: int = 32):
        """
        Добавление документов в индекс
        
        Args:
            texts: Список текстовых документов для добавления
            batch_size: Размер батча для обработки
        """
        if not texts:
            return
            
        print(f"Добавление {len(texts)} документов в индекс...")
        
        # Генерируем эмбеддинги батчами
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        # Объединяем все эмбеддинги
        embeddings = np.vstack(embeddings).astype('float32')
        
        # Добавляем в индекс
        if self.index.ntotal == 0:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
        else:
            self.index.add(embeddings)
        
        # Добавляем документы в список
        start_idx = len(self.documents)
        self.documents.extend(texts)
        print(f"Добавлено {len(texts)} документов. Всего документов: {len(self.documents)}")
        
        return list(range(start_idx, start_idx + len(texts)))
    
    def search(self, query: str, top_k: int = 3, threshold: float = 0.5) -> List[dict]:
        """
        Поиск наиболее релевантных документов
        
        Args:
            query: Поисковый запрос
            top_k: Количество возвращаемых результатов
            threshold: Порог косинусного сходства для фильтрации результатов
            
        Returns:
            Список словарей с результатами поиска
        """
        if not self.documents:
            return []
            
        # Получаем эмбеддинг запроса
        query_embedding = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True).astype('float32')
        
        # Ищем в индексе
        distances, indices = self.index.search(query_embedding, k=min(top_k, len(self.documents)))
        
        # Преобразуем результаты в список словарей
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.documents):
                # Преобразуем L2 расстояние в косинусное сходство (приблизительно)
                # Для нормализованных векторов: cosine_sim = 1 - distance^2 / 2
                similarity = 1 - (distance / 2.0)
                
                if similarity >= threshold:
                    results.append({
                        'text': self.documents[idx],
                        'score': float(similarity),
                        'index': int(idx)
                    })
        
        # Сортируем по убыванию релевантности
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def remove_document(self, doc_index: int):
        """Удаление документа по индексу"""
        if 0 <= doc_index < len(self.documents):
            # В FAISS нельзя удалять элементы напрямую, создаем новый индекс без удаляемого документа
            print(f"Удаление документа с индексом {doc_index}...")
            
            # Удаляем документ из списка
            del self.documents[doc_index]
            
            # Перестраиваем индекс
            if self.documents:
                # Генерируем эмбеддинги для оставшихся документов
                embeddings = self.model.encode(self.documents, show_progress_bar=True, convert_to_numpy=True)
                
                # Создаем новый индекс
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(embeddings.astype('float32'))
            else:
                # Если документов не осталось, создаем пустой индекс
                self.index = faiss.IndexFlatL2(self.dimension)
            
            print(f"Документ удален. Осталось документов: {len(self.documents)}")
            return True
        return False
    
    def clear_index(self):
        """Очистка индекса и удаление всех документов"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        print("Индекс и документы очищены")


def test_vector_search():
    """Тестирование работы векторного поиска"""
    # Пример использования
    vs = VectorSearch()
    
    # Добавляем тестовые документы
    test_docs = [
        "Рустам — фитнес-тренер и программист из села Завьялово.",
        "Елена Олеговна Дочия родилась 20 декабря 1984 года.",
        "Сын Рамиль родился 29 июня 2024 года.",
        "Пасынок Леон родился 17 июля 2011 года.",
        "Падчерица Артемида родилась 11 февраля 2009 года.",
        "Отец Ильдар Зинурович Исмагилов родился 12 июля 1964 года.",
        "Мать Людмила Александровна Исмагилова родилась 16 января 1965 года."
    ]
    
    vs.add_documents(test_docs)
    
    # Поиск
    query = "Когда родился Рамиль?"
    print(f"\nПоиск: {query}")
    results = vs.search(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [Сходство: {result['score']:.3f}] {result['text']}")
    
    # Сохраняем индекс
    vs.save_index()


if __name__ == "__main__":
    test_vector_search()
