#!/usr/bin/env bash
set -euo pipefail

echo "=== START load_mimic.sh ==="

# 1) Docker host gateway
HOST_IP=$(ip route | awk '/default/ {print $3}')
echo "→ Host gateway: $HOST_IP"

# 2) Postgres superuser creds
export PGHOST="$HOST_IP"
export PGPORT=5433
export PGUSER="postgres"
export PGPASSWORD="Iv7bahqu"
export PGDATABASE="mimic"
echo "→ Connecting to $PGUSER@$PGHOST:$PGPORT/$PGDATABASE"

# quick connectivity check
if ! PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "SELECT 1;" &>/dev/null; then
  echo "‼️  Cannot connect to Postgres — aborting" >&2
  exit 1
fi

# 3) Ensure the schema exists
echo "→ Ensuring schema mimiciii exists"
psql -q <<'EOF'
CREATE SCHEMA IF NOT EXISTS mimiciii AUTHORIZATION postgres;
EOF

# 4) Gather your CSV.GZ files in the correct folder
SRC_DIR="/home/graph-winit/physionet.org/files/mimiciii/1.4/"
echo "→ Scanning $SRC_DIR for .csv.gz files"
mapfile -t GZFILES < <(find "$SRC_DIR" -maxdepth 1 -type f -name '*.csv.gz')
echo "→ Found ${#GZFILES[@]} files"

if [[ ${#GZFILES[@]} -eq 0 ]]; then
  echo "‼️  No files found in $SRC_DIR; check your path!" >&2
  exit 1
fi

# 5) Loop through and load each
for GZ in "${GZFILES[@]}"; do
  BASENAME=$(basename "$GZ" .csv.gz)
  TBL=$(echo "$BASENAME" | tr '[:upper:]' '[:lower:]')
  echo "----------------------------------------"
  echo "→ Loading $BASENAME → mimiciii.$TBL"

#   read header
  echo "   → Reading header from $GZ"
  HEADER=$(zcat "$GZ" 2>/dev/null | head -n1)
  echo "$HEADER"
  
  if [[ -z "$HEADER" ]]; then
    echo "   ✗ ERROR: header is empty or unreadable for $GZ" >&2
    continue
  fi
  echo "   → Header: $HEADER"
  IFS=',' read -ra COLS <<<"$HEADER"


  # build CREATE TABLE if needed
  CREATE_SQL="CREATE TABLE IF NOT EXISTS mimiciii.\"$TBL\" ("
  for col in "${COLS[@]}"; do
    # strip outer quotes, escape internal quotes
    cname=$(echo "$col" | sed -e 's/^"//' -e 's/"$//' -e 's/\"/\"\"/g')
    CREATE_SQL+="\"$cname\" TEXT,"
  done
  CREATE_SQL=${CREATE_SQL%,}; CREATE_SQL+=");"
  echo "   → CREATE_SQL: $CREATE_SQL"
  psql -q -c "$CREATE_SQL"
  echo "   → Table mimiciii.$TBL exists"

  # truncate old data
  echo "   → Truncating mimiciii.$TBL"
  psql -q -c "TRUNCATE TABLE mimiciii.\"$TBL\";"

  # bulk load
  echo "   → Copying data"
  if zcat "$GZ" | psql -q <<EOF
\copy mimiciii."$TBL" FROM stdin WITH (FORMAT csv, HEADER true);
EOF
  then
    echo "   ✓ Imported $BASENAME"
  else
    echo "   ✗ Failed $BASENAME" >&2
  fi
done

# 6) Grant read access to mimicuser
echo "→ Granting SELECT on mimiciii to mimicuser"
psql -q <<'EOF'
GRANT USAGE ON SCHEMA mimiciii TO mimicuser;
GRANT SELECT ON ALL TABLES IN SCHEMA mimiciii TO mimicuser;
ALTER DEFAULT PRIVILEGES IN SCHEMA mimiciii
  GRANT SELECT ON TABLES TO mimicuser;
EOF

echo "=== DONE load_mimic.sh ==="
