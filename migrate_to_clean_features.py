#!/usr/bin/env python3
"""
migrate_to_clean_features.py

Script de migración segura de artifacts/ (39 features) → artifacts_v2/ (14 features)

Este script:
1. Hace backup de artifacts/ → artifacts_deprecated/
2. Promueve artifacts_v2/ → artifacts/
3. Verifica la integridad de los artifacts
4. Reporta el estado de la migración

Uso:
    python migrate_to_clean_features.py [--dry-run] [--force]

Opciones:
    --dry-run    Muestra lo que haría sin hacer cambios
    --force      Fuerza la migración incluso si hay warnings
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse


def check_artifacts_integrity(artifacts_path: Path) -> dict:
    """Verifica la integridad de un directorio de artifacts."""

    result = {
        "valid": True,
        "path": str(artifacts_path),
        "exists": artifacts_path.exists(),
        "files": {},
        "errors": []
    }

    if not artifacts_path.exists():
        result["valid"] = False
        result["errors"].append(f"Directorio no existe: {artifacts_path}")
        return result

    # Check required files
    required_files = ["model_best.pt", "meta.json", "scaler.pkl"]

    for filename in required_files:
        filepath = artifacts_path / filename
        exists = filepath.exists()
        result["files"][filename] = {
            "exists": exists,
            "size": filepath.stat().st_size if exists else 0
        }

        if not exists:
            result["valid"] = False
            result["errors"].append(f"Archivo faltante: {filename}")

    # Parse meta.json if exists
    meta_path = artifacts_path / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            result["meta"] = {
                "n_features": len(meta.get("feature_cols", [])),
                "horizon": meta.get("horizon"),
                "seq_len": meta.get("seq_len"),
                "hidden": meta.get("hidden"),
                "features": meta.get("feature_cols", [])
            }

            # Count price-level features
            price_level_patterns = ['log_close', 'sma_', 'ema_', 'bb_m', 'bb_up', 'bb_dn', 'fib_r_', 'fibext_']
            price_level_count = sum(
                1 for feat in meta.get("feature_cols", [])
                if any(p in feat for p in price_level_patterns)
            )
            result["meta"]["price_level_features"] = price_level_count

        except Exception as e:
            result["errors"].append(f"Error parsing meta.json: {e}")

    return result


def print_artifacts_info(info: dict, label: str):
    """Imprime información sobre un conjunto de artifacts."""

    print(f"\n{'='*60}")
    print(f"{label}: {info['path']}")
    print(f"{'='*60}")

    if not info["exists"]:
        print("[X] Directorio no existe")
        return

    if not info["valid"]:
        print("[X] Artifacts inválidos:")
        for error in info["errors"]:
            print(f"  - {error}")
        return

    print("[OK] Archivos:")
    for filename, file_info in info["files"].items():
        status = "[OK]" if file_info["exists"] else "[X]"
        size_kb = file_info["size"] / 1024
        print(f"  {status} {filename} ({size_kb:.1f} KB)")

    if "meta" in info:
        meta = info["meta"]
        print(f"\n[OK] Configuración:")
        print(f"  - Features: {meta['n_features']}")
        print(f"  - Horizon: {meta['horizon']}")
        print(f"  - Seq len: {meta['seq_len']}")
        print(f"  - Hidden: {meta['hidden']}")
        print(f"  - Price-level features: {meta['price_level_features']}")

        if meta['price_level_features'] > 0:
            print(f"    [WARN] {meta['price_level_features']} price-level features detectadas")
        else:
            print(f"    [OK] GOOD: 0 price-level features (price-invariant)")


def migrate(dry_run: bool = False, force: bool = False) -> bool:
    """
    Ejecuta la migración.

    Returns:
        True si la migración fue exitosa, False en caso contrario
    """

    print("\n" + "="*60)
    print("MIGRACIÓN A FEATURES LIMPIAS")
    print("="*60)

    if dry_run:
        print("\n[DRY-RUN] MODO DRY-RUN (no se haran cambios reales)")

    # Paths
    old_artifacts = Path("artifacts")
    new_artifacts = Path("artifacts_v2")
    deprecated_artifacts = Path("artifacts_deprecated")

    # Check current state
    print("\n1. Verificando estado actual...")

    old_info = check_artifacts_integrity(old_artifacts)
    new_info = check_artifacts_integrity(new_artifacts)

    print_artifacts_info(old_info, "ARTIFACTS ACTUALES (VIEJO)")
    print_artifacts_info(new_info, "ARTIFACTS NUEVOS (LIMPIO)")

    # Validation
    print("\n2. Validando migración...")

    issues = []

    if not old_info["exists"]:
        print("  [INFO] artifacts/ no existe (primer setup)")
    elif not old_info["valid"]:
        issues.append("artifacts/ actual está corrupto")

    if not new_info["exists"]:
        issues.append("artifacts_v2/ no existe - necesitas entrenar el modelo limpio primero")
    elif not new_info["valid"]:
        issues.append("artifacts_v2/ está corrupto")

    if deprecated_artifacts.exists():
        issues.append(f"artifacts_deprecated/ ya existe - muévelo o bórralo primero")

    # Check if new artifacts are actually clean
    if new_info.get("meta", {}).get("price_level_features", 0) > 0:
        warning = f"artifacts_v2/ tiene {new_info['meta']['price_level_features']} price-level features"
        if not force:
            issues.append(warning + " (usa --force para ignorar)")
        else:
            print(f"  [WARN] {warning} (continuando con --force)")

    if issues:
        print("\n[X] Problemas detectados:")
        for issue in issues:
            print(f"  - {issue}")

        if not force:
            print("\n[X] Migración abortada. Resuelve los problemas o usa --force")
            return False

    print("  [OK] Validación exitosa")

    # Execute migration
    print("\n3. Ejecutando migración...")

    steps = []

    if old_artifacts.exists():
        steps.append(f"  1. Backup: artifacts/ -> artifacts_deprecated/")
    steps.append(f"  2. Promover: artifacts_v2/ -> artifacts/")

    print("\nPasos a ejecutar:")
    for step in steps:
        print(step)

    if dry_run:
        print("\n[OK] DRY-RUN completado (no se hicieron cambios)")
        return True

    # Confirm
    print("\n[WARN] ESTA OPERACIÓN MOVERÁ ARCHIVOS")
    response = input("¿Continuar? (yes/no): ").strip().lower()

    if response not in ['yes', 'y', 'si', 's']:
        print("[X] Migración cancelada por el usuario")
        return False

    # Do the migration
    try:
        # Step 1: Backup old if exists
        if old_artifacts.exists():
            print(f"\n  Moviendo {old_artifacts} -> {deprecated_artifacts}...")
            shutil.move(str(old_artifacts), str(deprecated_artifacts))
            print("  [OK] Backup completado")

        # Step 2: Promote new
        print(f"\n  Moviendo {new_artifacts} -> {old_artifacts}...")
        shutil.move(str(new_artifacts), str(old_artifacts))
        print("  [OK] Promoción completada")

        print("\n" + "="*60)
        print("[OK] MIGRACIÓN COMPLETADA EXITOSAMENTE")
        print("="*60)

        # Verify new state
        final_info = check_artifacts_integrity(old_artifacts)
        print_artifacts_info(final_info, "ARTIFACTS ACTUALES (NUEVO)")

        print("\n[NEXT] Próximos pasos:")
        print("  1. Verificar que scripts usan las nuevas features:")
        print("     python test_feature_inspection.py")
        print("  2. Probar inferencia:")
        print("     python example_multi_horizon.py")
        print("  3. Verificar que el temporal lag se eliminó")

        return True

    except Exception as e:
        print(f"\n[X] ERROR durante la migración: {e}")
        print("\n[WARN] ESTADO INCONSISTENTE - revisa manualmente los directorios")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrar de artifacts/ (39 features) a artifacts_v2/ (14 features limpias)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar lo que se haría sin hacer cambios"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar migración incluso con warnings"
    )

    args = parser.parse_args()

    success = migrate(dry_run=args.dry_run, force=args.force)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
