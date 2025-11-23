#!/usr/bin/env python3
"""
PANTO Face Database Reader
Herramienta para visualizar y gestionar la base de datos de rostros

Uso:
    python read_database.py                    # Ver estadísticas
    python read_database.py --list             # Listar todas las personas
    python read_database.py --person p_xxxxx   # Ver detalles de una persona
    python read_database.py --update p_xxxxx "Juan Perez"  # Actualizar nombre
    python read_database.py --export persons.csv           # Exportar a CSV
    python read_database.py --stats                        # Estadísticas avanzadas
"""

import sqlite3
import argparse
import sys
import csv
from datetime import datetime
from pathlib import Path
import numpy as np

class FaceDatabase:
    def __init__(self, db_path="faces_v3.db"):
        self.db_path = db_path
        if not Path(db_path).exists():
            print(f"❌ Base de datos no encontrada: {db_path}")
            sys.exit(1)
        
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
    
    # ==================== ESTADÍSTICAS ====================
    
    def get_stats(self):
        """Obtener estadísticas generales"""
        stats = {}
        
        # Total de personas
        self.cursor.execute("SELECT COUNT(*) FROM persons")
        stats['total_persons'] = self.cursor.fetchone()[0]
        
        # Total de embeddings
        self.cursor.execute("SELECT COUNT(*) FROM face_embeddings")
        stats['total_embeddings'] = self.cursor.fetchone()[0]
        
        # Persona con más rostros
        self.cursor.execute("""
            SELECT p.person_id, p.name, p.total_faces 
            FROM persons p 
            ORDER BY p.total_faces DESC 
            LIMIT 1
        """)
        row = self.cursor.fetchone()
        if row:
            stats['most_faces'] = {
                'id': row[0],
                'name': row[1],
                'count': row[2]
            }
        
        # Calidad promedio
        self.cursor.execute("SELECT AVG(quality_score) FROM face_embeddings")
        stats['avg_quality'] = self.cursor.fetchone()[0] or 0.0
        
        # Fecha del primer registro
        self.cursor.execute("SELECT MIN(first_seen) FROM persons")
        first = self.cursor.fetchone()[0]
        if first:
            stats['first_seen'] = datetime.fromtimestamp(first / 1e9).strftime('%Y-%m-%d %H:%M:%S')
        
        # Fecha del último registro
        self.cursor.execute("SELECT MAX(last_seen) FROM persons")
        last = self.cursor.fetchone()[0]
        if last:
            stats['last_seen'] = datetime.fromtimestamp(last / 1e9).strftime('%Y-%m-%d %H:%M:%S')
        
        return stats
    
    def print_stats(self):
        """Imprimir estadísticas en consola"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS DE LA BASE DE DATOS")
        print("="*60)
        print(f"👥 Total de personas:        {stats['total_persons']}")
        print(f"📸 Total de embeddings:      {stats['total_embeddings']}")
        print(f"⭐ Calidad promedio:         {stats['avg_quality']:.2f}")
        
        if 'most_faces' in stats:
            mf = stats['most_faces']
            print(f"🏆 Más rostros guardados:    {mf['name']} ({mf['count']} rostros)")
        
        if 'first_seen' in stats:
            print(f"📅 Primer registro:          {stats['first_seen']}")
        
        if 'last_seen' in stats:
            print(f"📅 Último registro:          {stats['last_seen']}")
        
        print("="*60 + "\n")
    
    # ==================== LISTAR PERSONAS ====================
    
    def list_persons(self, limit=50):
        """Listar todas las personas"""
        self.cursor.execute("""
            SELECT 
                person_id,
                name,
                total_faces,
                best_quality,
                datetime(first_seen / 1000000000, 'unixepoch', 'localtime') as first_seen,
                datetime(last_seen / 1000000000, 'unixepoch', 'localtime') as last_seen
            FROM persons
            ORDER BY last_seen DESC
            LIMIT ?
        """, (limit,))
        
        rows = self.cursor.fetchall()
        
        if not rows:
            print("❌ No hay personas registradas\n")
            return
        
        print("\n" + "="*120)
        print(f"{'ID':<15} {'Nombre':<20} {'Rostros':<8} {'Calidad':<8} {'Primera vez':<20} {'Última vez':<20}")
        print("="*120)
        
        for row in rows:
            print(f"{row['person_id']:<15} {row['name']:<20} {row['total_faces']:<8} "
                  f"{row['best_quality']:<8.2f} {row['first_seen']:<20} {row['last_seen']:<20}")
        
        print("="*120)
        print(f"Total: {len(rows)} personas\n")
    
    # ==================== DETALLES DE PERSONA ====================
    
    def get_person_details(self, person_id):
        """Obtener detalles de una persona específica"""
        self.cursor.execute("""
            SELECT 
                person_id,
                name,
                total_faces,
                best_quality,
                datetime(first_seen / 1000000000, 'unixepoch', 'localtime') as first_seen,
                datetime(last_seen / 1000000000, 'unixepoch', 'localtime') as last_seen,
                notes
            FROM persons
            WHERE person_id = ?
        """, (person_id,))
        
        person = self.cursor.fetchone()
        
        if not person:
            print(f"❌ Persona no encontrada: {person_id}\n")
            return
        
        print("\n" + "="*60)
        print(f"👤 DETALLES DE PERSONA: {person['name']}")
        print("="*60)
        print(f"ID:              {person['person_id']}")
        print(f"Nombre:          {person['name']}")
        print(f"Total rostros:   {person['total_faces']}")
        print(f"Mejor calidad:   {person['best_quality']:.2f}")
        print(f"Primera vez:     {person['first_seen']}")
        print(f"Última vez:      {person['last_seen']}")
        if person['notes']:
            print(f"Notas:           {person['notes']}")
        print("="*60)
        
        # Obtener embeddings
        self.cursor.execute("""
            SELECT 
                embedding_id,
                quality_score,
                age,
                gender,
                gender_confidence,
                emotion,
                emotion_confidence,
                datetime(captured_at / 1000000000, 'unixepoch', 'localtime') as captured_at
            FROM face_embeddings
            WHERE person_id = ?
            ORDER BY quality_score DESC
            LIMIT 10
        """, (person_id,))
        
        embeddings = self.cursor.fetchall()
        
        if embeddings:
            print(f"\n📸 Últimos {len(embeddings)} embeddings (ordenados por calidad):")
            print("-"*120)
            print(f"{'ID':<8} {'Calidad':<8} {'Edad':<6} {'Género':<10} {'Conf':<6} {'Emoción':<12} {'Conf':<6} {'Capturado':<20}")
            print("-"*120)
            
            for emb in embeddings:
                print(f"{emb['embedding_id']:<8} {emb['quality_score']:<8.2f} "
                      f"{emb['age'] or 'N/A':<6} {emb['gender'] or 'N/A':<10} "
                      f"{emb['gender_confidence'] or 0:.2f}   "
                      f"{emb['emotion'] or 'N/A':<12} {emb['emotion_confidence'] or 0:.2f}   "
                      f"{emb['captured_at']:<20}")
            print("-"*120)
        
        print()
    
    # ==================== ACTUALIZAR NOMBRE ====================
    
    def update_name(self, person_id, new_name):
        """Actualizar nombre de una persona"""
        self.cursor.execute("SELECT person_id FROM persons WHERE person_id = ?", (person_id,))
        if not self.cursor.fetchone():
            print(f"❌ Persona no encontrada: {person_id}\n")
            return False
        
        self.cursor.execute("UPDATE persons SET name = ? WHERE person_id = ?", (new_name, person_id))
        self.conn.commit()
        
        print(f"✅ Nombre actualizado: {person_id} -> {new_name}\n")
        return True
    
    # ==================== EXPORTAR A CSV ====================
    
    def export_to_csv(self, output_file):
        """Exportar personas a CSV"""
        self.cursor.execute("""
            SELECT 
                person_id,
                name,
                total_faces,
                best_quality,
                datetime(first_seen / 1000000000, 'unixepoch', 'localtime') as first_seen,
                datetime(last_seen / 1000000000, 'unixepoch', 'localtime') as last_seen
            FROM persons
            ORDER BY last_seen DESC
        """)
        
        rows = self.cursor.fetchall()
        
        if not rows:
            print("❌ No hay datos para exportar\n")
            return
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['person_id', 'name', 'total_faces', 'best_quality', 'first_seen', 'last_seen'])
            
            for row in rows:
                writer.writerow([
                    row['person_id'],
                    row['name'],
                    row['total_faces'],
                    row['best_quality'],
                    row['first_seen'],
                    row['last_seen']
                ])
        
        print(f"✅ Exportado {len(rows)} personas a: {output_file}\n")
    
    # ==================== ESTADÍSTICAS AVANZADAS ====================
    
    def advanced_stats(self):
        """Estadísticas avanzadas"""
        print("\n" + "="*60)
        print("📈 ESTADÍSTICAS AVANZADAS")
        print("="*60)
        
        # Distribución de edad
        self.cursor.execute("""
            SELECT age, COUNT(*) as count 
            FROM face_embeddings 
            WHERE age IS NOT NULL 
            GROUP BY age 
            ORDER BY count DESC 
            LIMIT 5
        """)
        ages = self.cursor.fetchall()
        if ages:
            print("\n🎂 Edades más comunes:")
            for age in ages:
                print(f"   {age['age']} años: {age['count']} detecciones")
        
        # Distribución de género
        self.cursor.execute("""
            SELECT gender, COUNT(*) as count 
            FROM face_embeddings 
            WHERE gender IS NOT NULL 
            GROUP BY gender
        """)
        genders = self.cursor.fetchall()
        if genders:
            print("\n👥 Distribución de género:")
            for g in genders:
                print(f"   {g['gender']}: {g['count']} detecciones")
        
        # Emociones más comunes
        self.cursor.execute("""
            SELECT emotion, COUNT(*) as count 
            FROM face_embeddings 
            WHERE emotion IS NOT NULL AND emotion != 'Unknown'
            GROUP BY emotion 
            ORDER BY count DESC 
            LIMIT 5
        """)
        emotions = self.cursor.fetchall()
        if emotions:
            print("\n😊 Emociones más detectadas:")
            for emo in emotions:
                print(f"   {emo['emotion']}: {emo['count']} veces")
        
        # Calidad por persona
        self.cursor.execute("""
            SELECT p.name, AVG(f.quality_score) as avg_quality, COUNT(*) as count
            FROM persons p
            JOIN face_embeddings f ON p.person_id = f.person_id
            GROUP BY p.person_id
            ORDER BY avg_quality DESC
            LIMIT 5
        """)
        qualities = self.cursor.fetchall()
        if qualities:
            print("\n⭐ Personas con mejor calidad de rostros:")
            for q in qualities:
                print(f"   {q['name']}: {q['avg_quality']:.2f} ({q['count']} rostros)")
        
        print("="*60 + "\n")
    
    # ==================== BÚSQUEDA ====================
    
    def search_by_name(self, query):
        """Buscar personas por nombre"""
        self.cursor.execute("""
            SELECT 
                person_id,
                name,
                total_faces,
                best_quality
            FROM persons
            WHERE name LIKE ?
            ORDER BY total_faces DESC
        """, (f"%{query}%",))
        
        rows = self.cursor.fetchall()
        
        if not rows:
            print(f"❌ No se encontraron personas con nombre: {query}\n")
            return
        
        print(f"\n🔍 Resultados para '{query}':")
        print("-"*60)
        for row in rows:
            print(f"{row['person_id']:<15} {row['name']:<20} ({row['total_faces']} rostros, calidad: {row['best_quality']:.2f})")
        print("-"*60 + "\n")
    
    # ==================== LIMPIAR DUPLICADOS ====================
    
    def clean_duplicates(self, similarity_threshold=0.95):
        """Identificar posibles duplicados (requiere análisis manual)"""
        print("⚠️  Esta función requiere comparación de embeddings.")
        print("    Por ahora, revisa manualmente personas con nombres similares.\n")
        
        self.cursor.execute("""
            SELECT name, COUNT(*) as count 
            FROM persons 
            GROUP BY name 
            HAVING count > 1
            ORDER BY count DESC
        """)
        
        dupes = self.cursor.fetchall()
        
        if dupes:
            print("🔍 Posibles duplicados por nombre:")
            for d in dupes:
                print(f"   '{d['name']}': {d['count']} personas")
            print()
        else:
            print("✅ No se encontraron nombres duplicados\n")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='PANTO Face Database Reader')
    parser.add_argument('--db', default='faces_v3.db', help='Ruta a la base de datos')
    parser.add_argument('--list', action='store_true', help='Listar todas las personas')
    parser.add_argument('--person', type=str, help='Ver detalles de una persona (ID)')
    parser.add_argument('--update', nargs=2, metavar=('ID', 'NAME'), help='Actualizar nombre')
    parser.add_argument('--export', type=str, metavar='FILE', help='Exportar a CSV')
    parser.add_argument('--stats', action='store_true', help='Estadísticas avanzadas')
    parser.add_argument('--search', type=str, help='Buscar por nombre')
    parser.add_argument('--clean', action='store_true', help='Identificar duplicados')
    parser.add_argument('--limit', type=int, default=50, help='Límite de resultados (default: 50)')
    
    args = parser.parse_args()
    
    db = FaceDatabase(args.db)
    
    if args.list:
        db.list_persons(args.limit)
    elif args.person:
        db.get_person_details(args.person)
    elif args.update:
        db.update_name(args.update[0], args.update[1])
    elif args.export:
        db.export_to_csv(args.export)
    elif args.stats:
        db.advanced_stats()
    elif args.search:
        db.search_by_name(args.search)
    elif args.clean:
        db.clean_duplicates()
    else:
        # Default: mostrar estadísticas básicas
        db.print_stats()
        print("💡 Usa --help para ver todas las opciones disponibles\n")


if __name__ == "__main__":
    main()