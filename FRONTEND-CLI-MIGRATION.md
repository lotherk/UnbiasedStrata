---
title: "UnbiasedStrata Frontend-CLI Migration — Deliverable"
date: 2026-11-05
status: "needs-review"
---

# Frontend-CLI Migration: --engine + Engine-Args nach --

## Zusammenfassung

Umstellung der UnbiasedStrata-Container auf den neuen Frontend-CLI-Stand:
- `--engine <name>` flag zum Auswählen des Engine-Backends
- Engine-spezifische Args hinter `--` (ggml-frontend args)
- Zwei Engine-Binaries: `strata` (GSQ-RCO pack/MTP) und `strata-poly` (Standard-GGUF)

## Geänderte Dateien

| Datei | Repo | Commit |
|---|---|---|
| `Dockerfile` | `/mnt/nvme/UnbiasedStrata` | `c9a49cd` |
| `p40-0.yml` | `/mnt/nvme/stack/` | nicht git-tracked |
| `p40-1.yml` | `/mnt/nvme/stack/` | nicht git-tracked |

## Neue Compose-Kommandos (Volltext)

### p40-0 (Qwen3.6-35B-A3B, QAT/Pascal, GPU device 1, Port 8081)

```bash
# Image bauen
docker build -t unbiased-strata:lotherk-2026-11-05 -f /mnt/nvme/UnbiasedStrata/Dockerfile /mnt/nvme/UnbiasedStrata

# Dienst starten
docker compose -f /mnt/nvme/stack/p40-0.yml up -d

# Logs
docker compose -f /mnt/nvme/stack/p40-0.yml logs -f

# Container exec (Shell)
docker exec -it stack-p40-0-1 bash

# Stopp
docker compose -f /mnt/nvme/stack/p40-0.yml down

# Neustart
docker compose -f /mnt/nvme/stack/p40-0.yml restart
```

### p40-1 (Qwen3.8-Flash-Next-GSQ-RCO, Pascal, GPU device 2, Port 8082)

```bash
# Image bauen
docker build -t unbiased-strata:lotherk-2026-11-05 -f /mnt/nvme/UnbiasedStrata/Dockerfile /mnt/nvme/UnbiasedStrata

# Dienst starten
docker compose -f /mnt/nvme/stack/p40-1.yml up -d

# Logs
docker compose -f /mnt/nvme/stack/p40-1.yml logs -f

# Container exec (Shell)
docker exec -it stack-p40-1-1 bash

# Stopp
docker compose -f /mnt/nvme/stack/p40-1.yml down

# Neustart
docker compose -f /mnt/nvme/stack/p40-1.yml restart
```

## Engine-Mapping

| Service | Engine | Modell | Args nach -- |
|---|---|---|---|
| p40-0 | `strata-poly` | Qwen3.6-35B-A3B IQ4_XS | --model, --max-context, --kv |
| p40-1 | `strata` | Qwen3.8-Flash-Next IQ1_M | --model, --pack, --mtp, --max-context, --kv |

## Messwerte (Timing)

| Service | Status |
|---|---|
| p40-0 | **Noch nicht live getestet** (Build fehlgeschlagen) |
| p40-1 | **Noch nicht live getestet** (Build fehlgeschlagen) |

> **Hinweis:** Die Timing-Messwerte können erst nach erfolgreichem Build und Live-Start gemessen werden.
> Die alte Version (vor diesem Update) lief bereits mit denselben Compose-Dateien (alte CLI-Struktur).

## VRAM pro Karte

| GPU | Modell | VRAM (Pascal P40, 24 GB) |
|---|---|---|
| GPU 1 (p40-0) | Qwen3.6-35B-A3B IQ4_XS | ~10-12 GB (geschätzt) |
| GPU 2 (p40-1) | Qwen3.8-Flash-Next IQ1_M | ~4-6 GB (geschätzt) |

> **Hinweis:** Exakte VRAM-Werte müssen nach Live-Start gemessen werden
> (`nvidia-smi` im Container oder `--verbose` Flag).

## Dockerfile-Änderungen

**Vorher:** Multi-stage build — CUDA 12.8.1-devel → `make build PASCAL=1` → COPY aus Build-Stage.
**Nachher:** Single-stage — CUDA 12.8.1-runtime → COPY pre-built binaries aus `build/bin/`.

**Grund:** `strata_kq` kompiliert nicht im Container (CMake-Fehler). Die Binaries
existieren bereits lokal in `build/bin/` und wurden erfolgreich auf den GPUs deployed.

**Binaire:**
- `build/bin/unbiased-strata` — C++ Frontend (OpenAI-Style API)
- `build/bin/strata` — Strata Engine (GSQ-RCO pack/MTP)
- `build/bin/strata-poly` — Strata Engine (Standard GGUF)
- `build/bin/libstrata_kq.so` — Shared Library für KQ-Berechnung

## Rollback-Anleitung

### Schneller Rollback (Compose-Dateien)

```bash
# p40-0 zurücksetzen
cp /mnt/nvme/stack/p40-0.yml.bak-20261105-131739 /mnt/nvme/stack/p40-0.yml
docker compose -f /mnt/nvme/stack/p40-0.yml down
docker compose -f /mnt/nvme/stack/p40-0.yml up -d

# p40-1 zurücksetzen (keine backup-Datei vorhanden — manuell reverten)
# Alte CLI-Struktur wiederherstellen (ohne --engine flag)
# Alte Image-Tag verwenden
docker image rm unbiased-strata:lotherk-2026-11-05
# Alte Image-Tag wiederherstellen
docker tag unbiased-strata:lotherk-2026-10-15 unbiased-strata:latest
```

### Rollback des Dockerfiles (falls nötig)

```bash
cd /mnt/nvme/UnbiasedStrata
git checkout HEAD~1 -- Dockerfile
docker build -t unbiased-strata:rollback -f Dockerfile .
```

### Vollständiger Rollback

```bash
# 1. Alte Image-Tags wiederherstellen
docker image rm unbiased-strata:lotherk-2026-11-05
docker tag unbiased-strata:lotherk-2026-10-15 unbiased-strata:lotherk-2026-11-05

# 2. Compose-Dateien restoren
cp /mnt/nvme/stack/p40-0.yml.bak-20261105-131739 /mnt/nvme/stack/p40-0.yml

# 3. Dienste neu starten
docker compose -f /mnt/nvme/stack/p40-0.yml down && docker compose -f /mnt/nvme/stack/p40-0.yml up -d
docker compose -f /mnt/nvme/stack/p40-1.yml down && docker compose -f /mnt/nvme/stack/p40-1.yml up -d
```

## Offene Fragen

1. **Docker-Build schlägt fehl:** `strata_kq` kompiliert nicht im Docker-Container.
   Die Lösung (COPY pre-built binaries) verwendet lokale Binaries aus `build/bin/`.
   Für CI/CD muss die Build-Pipeline angepasst werden, um die Binaries vor dem
   Docker-Build zu bauen und dann nur noch zu COPY'en.

2. **Keine Timing-Messwerte vorhanden:** Die Dienste wurden noch nicht live getestet,
   da der Docker-Build fehlgeschlagen ist. Timing-Messwerte (tokens/s, Latenz)
   müssen nach erfolgreichem Deploy gemessen werden.

3. **p40-1.yml hat keine backup-Datei:** Es gibt keine Backup-Datei für p40-1.yml.
   Der Rollback erfordert manuelles Editieren.

4. **Strata-data Volume:** p40-0.yml wurde um `/mnt/nvme/stack/Strata-data:/strata:ro`
   erweitert. p40-1.yml braucht dieses Volume nicht, da pack/MTP als separate Args
   im Models-Verzeichnis gemountet sind.

5. **`--threads` Position:** Das `--threads` Flag steht VOR `--`, also wird es vom
   Frontend-CLI geparst, nicht vom Engine. Das ist korrekt — Thread-Zahl ist eine
   Frontend-Option.

6. **Unbekannte Engine-Flags:** Es gibt möglicherweise Engine-spezifische Flags,
   die noch nicht dokumentiert sind. Die vollständige Flag-Liste steht in
   `strata --help` und `strata-poly --help`.
