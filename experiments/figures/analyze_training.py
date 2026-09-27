"""
Analisa o log de treino e sugere melhorias.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_training_log(log_path='log/local_reconstruction/training_log.csv'):
    """Analisa o log de treino."""
    
    print("="*80)
    print("ANÁLISE DO TREINO")
    print("="*80)
    
    # Carregar log
    df = pd.read_csv(log_path)
    
    print(f"\nTotal de epochs: {len(df)}")
    print(f"Learning rate inicial: {df['learning_rate'].iloc[0]}")
    print(f"Learning rate final: {df['learning_rate'].iloc[-1]}")
    
    # Encontrar melhor epoch
    best_epoch = df['val_loss'].idxmin() + 1
    best_val_loss = df['val_loss'].min()
    
    print(f"\nMelhor epoch: {best_epoch}")
    print(f"Melhor val loss: {best_val_loss:.4f}")
    print(f"Train loss no melhor epoch: {df.loc[best_epoch-1, 'train_loss']:.4f}")
    
    # Analisar overfitting
    final_train_loss = df['train_loss'].iloc[-1]
    final_val_loss = df['val_loss'].iloc[-1]
    
    print(f"\nLoss final:")
    print(f"  Train: {final_train_loss:.4f}")
    print(f"  Val: {final_val_loss:.4f}")
    print(f"  Gap: {final_val_loss - final_train_loss:.4f}")
    
    # Verificar quando começou a divergir
    train_at_best = df.loc[best_epoch-1, 'train_loss']
    val_at_best = df.loc[best_epoch-1, 'val_loss']
    
    print(f"\nNo melhor epoch:")
    print(f"  Train: {train_at_best:.4f}")
    print(f"  Val: {val_at_best:.4f}")
    print(f"  Gap: {val_at_best - train_at_best:.4f}")
    
    # Analisar instabilidade
    val_loss_std = df['val_loss'].std()
    val_loss_mean = df['val_loss'].mean()
    
    print(f"\nInstabilidade da val loss:")
    print(f"  Mean: {val_loss_mean:.4f}")
    print(f"  Std: {val_loss_std:.4f}")
    print(f"  CV: {val_loss_std/val_loss_mean:.2%}")
    
    # Verificar quando LR mudou
    lr_changes = df[df['learning_rate'] != df['learning_rate'].shift()].index.tolist()
    print(f"\nMudanças de learning rate nos epochs: {[i+1 for i in lr_changes]}")
    
    # Sugestões
    print("\n" + "="*80)
    print("DIAGNÓSTICO E SUGESTÕES")
    print("="*80)
    
    if final_val_loss > best_val_loss * 2:
        print("\n⚠️  PROBLEMA: Overfitting severo após melhor epoch")
        print("   A val loss aumentou drasticamente após o epoch", best_epoch)
        print("\n   Possíveis causas:")
        print("   1. Learning rate muito alto após o melhor epoch")
        print("   2. Modelo muito complexo para o dataset")
        print("   3. Falta de regularização")
        
        print("\n   Sugestões:")
        print("   1. Usar early stopping (parar no epoch", best_epoch + 10, ")")
        print("   2. Reduzir learning rate mais cedo (--lr_decay_step 15)")
        print("   3. Adicionar dropout ou weight decay maior")
        print("   4. Reduzir latent_dim (tentar 256 ou 384)")
        print("   5. Aumentar batch_size se possível")
    
    if val_loss_std / val_loss_mean > 0.3:
        print("\n⚠️  PROBLEMA: Treino muito instável")
        print("   A val loss varia muito entre epochs")
        
        print("\n   Sugestões:")
        print("   1. Reduzir learning rate inicial (--lr 0.0005)")
        print("   2. Usar learning rate scheduler mais suave")
        print("   3. Aumentar batch_size para estabilizar gradientes")
        print("   4. Verificar se há outliers no dataset")
    
    # Verificar se melhorou nos últimos epochs
    last_10_mean = df['val_loss'].iloc[-10:].mean()
    if last_10_mean > best_val_loss * 1.5:
        print("\n⚠️  PROBLEMA: Modelo não está melhorando")
        print("   Últimos 10 epochs têm val loss muito maior que o melhor")
        
        print("\n   Sugestões:")
        print("   1. Parar o treino e usar o melhor modelo (epoch", best_epoch, ")")
        print("   2. Retreinar com early stopping")
        print("   3. Ajustar hiperparâmetros")
    
    # Análise de componentes da loss
    print("\n" + "="*80)
    print("ANÁLISE DE COMPONENTES DA LOSS")
    print("="*80)
    
    best_cd = df.loc[best_epoch-1, 'val_cd_reconstruction']
    best_coverage = df.loc[best_epoch-1, 'val_coverage']
    
    final_cd = df['val_cd_reconstruction'].iloc[-1]
    final_coverage = df['val_coverage'].iloc[-1]
    
    print(f"\nNo melhor epoch:")
    print(f"  CD Reconstruction: {best_cd:.4f}")
    print(f"  Coverage: {best_coverage:.4f}")
    
    print(f"\nFinal:")
    print(f"  CD Reconstruction: {final_cd:.4f}")
    print(f"  Coverage: {final_coverage:.4f}")
    
    print(f"\nVariação:")
    print(f"  CD Reconstruction: +{(final_cd - best_cd)/best_cd*100:.1f}%")
    print(f"  Coverage: +{(final_coverage - best_coverage)/best_coverage*100:.1f}%")
    
    # Comando recomendado
    print("\n" + "="*80)
    print("COMANDO RECOMENDADO PARA RETREINO")
    print("="*80)
    
    print("\npython train_local_reconstruction.py \\")
    print("  --data_root data/DMC \\")
    print("  --num_points 2048 \\")
    print("  --batch_size 32 \\  # Aumentado para estabilidade")
    print("  --epochs 50 \\  # Reduzido, usar early stopping")
    print("  --lr 0.0005 \\  # Reduzido para estabilidade")
    print("  --latent_dim 384 \\  # Reduzido para evitar overfitting")
    print("  --predict_ratio 0.5 \\")
    print("  --coverage_weight 0.5 \\")
    print("  --lr_decay_step 20 \\  # Mais cedo")
    print("  --lr_decay_rate 0.5 \\")
    print("  --weight_decay 0.001 \\  # Aumentado para regularização")
    print("  --log_dir ./log/local_reconstruction_v2 \\")
    print("  --gpu 0")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    analyze_training_log()
