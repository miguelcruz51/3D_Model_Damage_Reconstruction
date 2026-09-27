"""
Análise de Estatísticas do Dataset DMC

Analisa:
- Número de componentes
- Número de versões por componente
- Partições danificadas vs não danificadas
- Estatísticas de pontos (danificados vs bons)
- Estatísticas de GT (intactos vs reconstrução)
"""

import os
import glob
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def analyze_dataset(data_root='data/DMC'):
    """Analisa o dataset completo."""
    
    print("="*80)
    print("ANÁLISE DO DATASET DMC")
    print("="*80)
    
    # Estruturas para armazenar estatísticas
    stats = {
        'components': [],
        'total_partitions': 0,
        'damaged_partitions': 0,
        'non_damaged_partitions': 0,
        'partitions_with_gt': 0,
        'total_damaged_points': 0,
        'total_intact_points': 0,
        'total_gt_reconstruction_points': 0,
        'total_gt_intact_points': 0,
        'component_stats': {}
    }
    
    # Iterar por componentes
    for comp_id in range(1, 17):
        comp_str = f"{comp_id:03d}"
        comp_path = os.path.join(data_root, comp_str, 'partitions')
        
        if not os.path.exists(comp_path):
            continue
        
        stats['components'].append(comp_str)
        
        # Estatísticas do componente
        comp_stats = {
            'versions': set(),
            'total_partitions': 0,
            'damaged_partitions': 0,
            'non_damaged_partitions': 0,
            'partitions_with_gt': 0,
            'damaged_points': 0,
            'intact_points': 0,
            'gt_reconstruction_points': 0,
            'gt_intact_points': 0
        }
        
        # Encontrar todos os arquivos .npz
        files = glob.glob(os.path.join(comp_path, '*.npz'))
        
        for filepath in files:
            filename = os.path.basename(filepath)
            
            # Pular arquivos GT
            if '_gt' in filename:
                continue
            
            # Parse filename: {comp}{version}{partition}{flag}.npz
            # Exemplo: 001001000201.npz
            if len(filename) < 16:
                continue
            
            version = filename[3:6]
            partition_num = filename[6:10]
            damage_flag = filename[10:12]
            
            comp_stats['versions'].add(version)
            comp_stats['total_partitions'] += 1
            stats['total_partitions'] += 1
            
            # Carregar dados
            try:
                data = np.load(filepath)
                points = data['points']
                labels = data['labels']
                
                # Contar pontos danificados e intactos
                damaged_count = np.sum(labels == 1)
                intact_count = np.sum(labels == 0)
                
                comp_stats['damaged_points'] += damaged_count
                comp_stats['intact_points'] += intact_count
                stats['total_damaged_points'] += damaged_count
                stats['total_intact_points'] += intact_count
                
                # Verificar se é danificada ou não
                if damage_flag == '01':
                    comp_stats['damaged_partitions'] += 1
                    stats['damaged_partitions'] += 1
                else:
                    comp_stats['non_damaged_partitions'] += 1
                    stats['non_damaged_partitions'] += 1
                
                # Verificar se tem GT
                gt_filepath = filepath.replace('.npz', '_gt.npz')
                if os.path.exists(gt_filepath):
                    comp_stats['partitions_with_gt'] += 1
                    stats['partitions_with_gt'] += 1
                    
                    # Carregar GT
                    gt_data = np.load(gt_filepath)
                    gt_labels = gt_data['labels']
                    
                    gt_recon_count = np.sum(gt_labels == 1)
                    gt_intact_count = np.sum(gt_labels == 0)
                    
                    comp_stats['gt_reconstruction_points'] += gt_recon_count
                    comp_stats['gt_intact_points'] += gt_intact_count
                    stats['total_gt_reconstruction_points'] += gt_recon_count
                    stats['total_gt_intact_points'] += gt_intact_count
                
            except Exception as e:
                print(f"Erro ao carregar {filename}: {e}")
                continue
        
        stats['component_stats'][comp_str] = comp_stats
    
    return stats


def print_statistics(stats):
    """Imprime estatísticas formatadas."""
    
    print("\n" + "="*80)
    print("ESTATÍSTICAS GERAIS")
    print("="*80)
    
    print(f"\nComponentes encontrados: {len(stats['components'])}")
    print(f"Componentes: {', '.join(stats['components'])}")
    
    print(f"\nTotal de partições: {stats['total_partitions']}")
    print(f"  Partições danificadas (flag='01'): {stats['damaged_partitions']} ({stats['damaged_partitions']/stats['total_partitions']*100:.1f}%)")
    print(f"  Partições não danificadas (flag='00'): {stats['non_damaged_partitions']} ({stats['non_damaged_partitions']/stats['total_partitions']*100:.1f}%)")
    print(f"  Partições com GT: {stats['partitions_with_gt']} ({stats['partitions_with_gt']/stats['total_partitions']*100:.1f}%)")
    
    print(f"\nEstatísticas de pontos (partições danificadas):")
    total_points = stats['total_damaged_points'] + stats['total_intact_points']
    print(f"  Total de pontos: {total_points:,}")
    print(f"  Pontos danificados (label=1): {stats['total_damaged_points']:,} ({stats['total_damaged_points']/total_points*100:.1f}%)")
    print(f"  Pontos intactos (label=0): {stats['total_intact_points']:,} ({stats['total_intact_points']/total_points*100:.1f}%)")
    
    if stats['partitions_with_gt'] > 0:
        print(f"\nEstatísticas de GT (partições com GT):")
        total_gt_points = stats['total_gt_reconstruction_points'] + stats['total_gt_intact_points']
        print(f"  Total de pontos GT: {total_gt_points:,}")
        print(f"  Pontos de reconstrução (label=1): {stats['total_gt_reconstruction_points']:,} ({stats['total_gt_reconstruction_points']/total_gt_points*100:.1f}%)")
        print(f"  Pontos intactos (label=0): {stats['total_gt_intact_points']:,} ({stats['total_gt_intact_points']/total_gt_points*100:.1f}%)")
    
    # Estatísticas por componente
    print("\n" + "="*80)
    print("ESTATÍSTICAS POR COMPONENTE")
    print("="*80)
    
    for comp_id in sorted(stats['component_stats'].keys()):
        comp_stats = stats['component_stats'][comp_id]
        
        print(f"\nComponente {comp_id}:")
        print(f"  Versões: {len(comp_stats['versions'])} ({', '.join(sorted(comp_stats['versions']))})")
        print(f"  Total de partições: {comp_stats['total_partitions']}")
        print(f"    Danificadas: {comp_stats['damaged_partitions']}")
        print(f"    Não danificadas: {comp_stats['non_damaged_partitions']}")
        print(f"    Com GT: {comp_stats['partitions_with_gt']}")
        
        if comp_stats['damaged_points'] + comp_stats['intact_points'] > 0:
            total_pts = comp_stats['damaged_points'] + comp_stats['intact_points']
            print(f"  Pontos danificados: {comp_stats['damaged_points']:,} ({comp_stats['damaged_points']/total_pts*100:.1f}%)")
            print(f"  Pontos intactos: {comp_stats['intact_points']:,} ({comp_stats['intact_points']/total_pts*100:.1f}%)")


def plot_statistics(stats, output_dir='data/DMC'):
    """Gera gráficos das estatísticas."""
    
    print("\n" + "="*80)
    print("GERANDO GRÁFICOS")
    print("="*80)
    
    # Preparar dados
    components = sorted(stats['component_stats'].keys())
    damaged_partitions = [stats['component_stats'][c]['damaged_partitions'] for c in components]
    non_damaged_partitions = [stats['component_stats'][c]['non_damaged_partitions'] for c in components]
    
    # Criar figura com múltiplos subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Estatísticas do Dataset DMC', fontsize=16, fontweight='bold')
    
    # Plot 1: Partições por componente
    ax = axes[0, 0]
    x = np.arange(len(components))
    width = 0.35
    ax.bar(x - width/2, damaged_partitions, width, label='Danificadas', color='#e74c3c')
    ax.bar(x + width/2, non_damaged_partitions, width, label='Não danificadas', color='#95a5a6')
    ax.set_xlabel('Componente', fontsize=12)
    ax.set_ylabel('Número de Partições', fontsize=12)
    ax.set_title('Partições por Componente', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Distribuição geral de partições
    ax = axes[0, 1]
    labels = ['Danificadas\n(com GT)', 'Não danificadas\n(sem GT)']
    sizes = [stats['damaged_partitions'], stats['non_damaged_partitions']]
    colors = ['#e74c3c', '#95a5a6']
    explode = (0.1, 0)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 11})
    ax.set_title('Distribuição de Partições', fontsize=14, fontweight='bold')
    
    # Plot 3: Pontos danificados vs intactos
    ax = axes[1, 0]
    labels = ['Pontos\nDanificados', 'Pontos\nIntactos']
    sizes = [stats['total_damaged_points'], stats['total_intact_points']]
    colors = ['#e74c3c', '#3498db']
    explode = (0.1, 0)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 11})
    ax.set_title('Distribuição de Pontos (Partições Danificadas)', fontsize=14, fontweight='bold')
    
    # Plot 4: Pontos GT
    ax = axes[1, 1]
    if stats['total_gt_reconstruction_points'] > 0:
        labels = ['Reconstrução\n(GT)', 'Intactos\n(GT)']
        sizes = [stats['total_gt_reconstruction_points'], stats['total_gt_intact_points']]
        colors = ['#2ecc71', '#3498db']
        explode = (0.1, 0)
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 11})
        ax.set_title('Distribuição de Pontos GT', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Sem dados GT', ha='center', va='center', fontsize=14)
        ax.set_title('Distribuição de Pontos GT', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar figura
    output_path = os.path.join(output_dir, 'dataset_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico salvo em: {output_path}")
    
    # Criar gráfico de barras detalhado por componente
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(components))
    width = 0.25
    
    damaged_pts = [stats['component_stats'][c]['damaged_points'] for c in components]
    intact_pts = [stats['component_stats'][c]['intact_points'] for c in components]
    gt_recon_pts = [stats['component_stats'][c]['gt_reconstruction_points'] for c in components]
    
    ax.bar(x - width, damaged_pts, width, label='Pontos Danificados', color='#e74c3c')
    ax.bar(x, intact_pts, width, label='Pontos Intactos', color='#3498db')
    ax.bar(x + width, gt_recon_pts, width, label='Pontos Reconstrução (GT)', color='#2ecc71')
    
    ax.set_xlabel('Componente', fontsize=14)
    ax.set_ylabel('Número de Pontos', fontsize=14)
    ax.set_title('Distribuição de Pontos por Componente', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=45)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'points_per_component.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico salvo em: {output_path}")


def save_statistics_csv(stats, output_dir='data/DMC'):
    """Salva estatísticas em CSV."""
    
    import csv
    
    # CSV geral
    csv_path = os.path.join(output_dir, 'dataset_statistics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Métrica', 'Valor'])
        writer.writerow(['Total de componentes', len(stats['components'])])
        writer.writerow(['Total de partições', stats['total_partitions']])
        writer.writerow(['Partições danificadas', stats['damaged_partitions']])
        writer.writerow(['Partições não danificadas', stats['non_damaged_partitions']])
        writer.writerow(['Partições com GT', stats['partitions_with_gt']])
        writer.writerow(['Total pontos danificados', stats['total_damaged_points']])
        writer.writerow(['Total pontos intactos', stats['total_intact_points']])
        writer.writerow(['Total pontos reconstrução GT', stats['total_gt_reconstruction_points']])
        writer.writerow(['Total pontos intactos GT', stats['total_gt_intact_points']])
    
    print(f"\nEstatísticas gerais salvas em: {csv_path}")
    
    # CSV por componente
    csv_path = os.path.join(output_dir, 'component_statistics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Componente', 'Versões', 'Total Partições', 'Danificadas', 'Não Danificadas',
            'Com GT', 'Pontos Danificados', 'Pontos Intactos', 'Pontos Reconstrução GT'
        ])
        
        for comp_id in sorted(stats['component_stats'].keys()):
            comp_stats = stats['component_stats'][comp_id]
            writer.writerow([
                comp_id,
                len(comp_stats['versions']),
                comp_stats['total_partitions'],
                comp_stats['damaged_partitions'],
                comp_stats['non_damaged_partitions'],
                comp_stats['partitions_with_gt'],
                comp_stats['damaged_points'],
                comp_stats['intact_points'],
                comp_stats['gt_reconstruction_points']
            ])
    
    print(f"Estatísticas por componente salvas em: {csv_path}")


def main():
    """Função principal."""
    
    data_root = 'data/DMC'
    
    # Analisar dataset
    stats = analyze_dataset(data_root)
    
    # Imprimir estatísticas
    print_statistics(stats)
    
    # Gerar gráficos
    plot_statistics(stats, data_root)
    
    # Salvar CSVs
    save_statistics_csv(stats, data_root)
    
    print("\n" + "="*80)
    print("ANÁLISE COMPLETA!")
    print("="*80)
    print(f"\nArquivos gerados em: {data_root}/")
    print("  - dataset_statistics.png")
    print("  - points_per_component.png")
    print("  - dataset_statistics.csv")
    print("  - component_statistics.csv")


if __name__ == '__main__':
    main()
