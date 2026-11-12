//! GGUF v2/v3 parser over a memory map.

use anyhow::{Context, bail};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::path::Path;

use crate::quant::GgmlType;

#[derive(Clone, Debug)]
pub enum Value {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), U64(u64), I64(i64),
    F32(f32), F64(f64), Bool(bool), Str(String), Arr(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64, Value::U16(v) => v as u64, Value::U32(v) => v as u64, Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64, Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64, Value::I64(v) if v >= 0 => v as u64,
            Value::Bool(v) => v as u64,
            _ => return None,
        })
    }
    pub fn as_f32(&self) -> Option<f32> {
        match *self { Value::F32(v) => Some(v), Value::F64(v) => Some(v as f32), _ => self.as_u64().map(|v| v as f32) }
    }
    pub fn as_str(&self) -> Option<&str> { if let Value::Str(s) = self { Some(s) } else { None } }
    pub fn as_arr(&self) -> Option<&[Value]> { if let Value::Arr(a) = self { Some(a) } else { None } }
}

#[derive(Clone, Debug)]
pub struct TensorInfo { pub name: String, pub dims: Vec<u64>, pub ty: u32, pub offset: u64, /** Shard index. */ pub file: usize }

pub struct Gguf {
    pub meta: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorInfo>,
    /// Each shard's map and data offset (one entry unless the model is split).
    files: Vec<(Mmap, usize)>,
}

struct Cur<'a> { b: &'a [u8], p: usize }

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        let end = self.p.checked_add(n).context("GGUF truncated")?;
        let s = self.b.get(self.p..end).context("GGUF truncated")?;
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> anyhow::Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into()?)) }
    fn u64(&mut self) -> anyhow::Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into()?)) }
    fn str(&mut self) -> anyhow::Result<String> {
        let n = usize::try_from(self.u64()?).context("GGUF string length out of range")?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn val(&mut self, t: u32) -> anyhow::Result<Value> {
        Ok(match t {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into()?)),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.str()?),
            9 => {
                let (et, n) = (self.u32()?, usize::try_from(self.u64()?).context("GGUF array length out of range")?);
                if n > self.b.len().saturating_sub(self.p) { bail!("GGUF array length exceeds remaining data"); }
                let mut a = Vec::with_capacity(n.min(1024));
                for _ in 0..n { a.push(self.val(et)?); }
                Value::Arr(a)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => bail!("unknown GGUF value type {t}"),
        })
    }
}

impl Gguf {
    /// Open a GGUF; for a split model (`split.count` > 1), pass the `-00001-of-0000N` shard and the others are
    /// opened from the same directory.
    pub fn open(path: &Path) -> anyhow::Result<Gguf> {
        let mut g = Self::open_one(path, 0)?;
        let n = g.meta.get("split.count").and_then(Value::as_u64).unwrap_or(1) as usize;
        if n > 1 {
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let tag = format!("-00001-of-{n:05}.gguf");
            let stem = name.strip_suffix(&tag).with_context(|| format!("{}: split model; open its first shard (*{tag})", path.display()))?;
            for i in 1..n {
                let s = Self::open_one(&path.with_file_name(format!("{stem}-{:05}-of-{n:05}.gguf", i + 1)), i)?;
                g.files.extend(s.files);
                for (k, t) in s.tensors { anyhow::ensure!(g.tensors.insert(k.clone(), t).is_none(), "tensor {k} in two shards"); }
            }
        }
        // A file without the n-gram (PLE) table takes it from the GGUF that `--ple` names, as one more shard.
        const PLE: &str = "per_layer_token_embd.weight";
        if g.meta.keys().any(|k| k.ends_with(".ple.layers")) && let Some(p) = std::env::var_os("STRATA_PLE") {
            let p = Path::new(&p);
            if g.tensors.contains_key(PLE) {
                eprintln!("--ple: {} has its own n-gram table, which it keeps", path.display());
            } else {
                let mut s = Self::open_one(p, g.files.len())?;
                let t = s.tensors.remove(PLE).with_context(|| format!("{}: no tensor {PLE}", p.display()))?;
                // The table is indexed by hashes of token n-grams: another model's hash constants read other rows.
                for k in ["layer_multipliers", "head_offsets", "head_vocab_sizes", "ngram_size", "heads_per_ngram"] {
                    let v = |m: &BTreeMap<String, Value>| m.iter().find(|(n, _)| n.ends_with(&format!(".ple.{k}"))).map(|(_, v)| format!("{v:?}"));
                    anyhow::ensure!(v(&s.meta).is_none_or(|x| Some(x) == v(&g.meta)), "{}: its n-gram table is for another model (ple.{k} differs)", p.display());
                }
                // Nothing in a file without the table says which rounding of it the file was made and measured with.
                eprintln!("n-gram table: {} from {} (--ple). The answers equal those of the model's full file only if the two tables \
                    are the same bytes; another quantization of the table changes some tokens.",
                    GgmlType::from_u32(t.ty).map_or("?", |ty| ty.name()), p.display());
                g.files.extend(s.files);
                g.tensors.insert(PLE.into(), t);
            }
        }
        Ok(g)
    }

    fn open_one(path: &Path, file: usize) -> anyhow::Result<Gguf> {
        let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&f)? };
        let mut c = Cur { b: &mmap, p: 0 };
        if c.take(4)? != b"GGUF" { bail!("{}: not a GGUF file", path.display()); }
        let ver = c.u32()?;
        if !(2..=3).contains(&ver) { bail!("unsupported GGUF version {ver}"); }
        let (nt, nkv) = (c.u64()?, c.u64()?);
        let mut meta = BTreeMap::new();
        for _ in 0..nkv {
            let k = c.str()?;
            let t = c.u32()?;
            meta.insert(k, c.val(t)?);
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..nt {
            let name = c.str()?;
            let nd = c.u32()?;
            let dims = (0..nd).map(|_| c.u64()).collect::<anyhow::Result<Vec<_>>>()?;
            let (ty, offset) = (c.u32()?, c.u64()?);
            tensors.insert(name.clone(), TensorInfo { name, dims, ty, offset, file });
        }
        let align = usize::try_from(meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32))
            .context("GGUF alignment out of range")?;
        let data_offset = c.p.checked_next_multiple_of(align).context("invalid GGUF alignment")?;
        // A cut-off download keeps a valid header; catch it here, not as a slice panic in `bytes`.
        for t in tensors.values() {
            let Ok(ty) = GgmlType::from_u32(t.ty) else { continue };  // Weights::validate reports it
            let n = t.dims.iter().skip(1).try_fold(ty.row_bytes(t.dims.first().copied().unwrap_or(0)) as u64, |a, &d| a.checked_mul(d));
            if n.and_then(|n| n.checked_add(t.offset)).is_none_or(|e| e > (mmap.len() as u64).saturating_sub(data_offset as u64)) {
                bail!("{}: GGUF truncated (tensor {} lies past the end of the file; re-download it)", path.display(), t.name);
            }
        }
        Ok(Gguf { meta, tensors, files: vec![(mmap, data_offset)] })
    }

    pub fn get(&self, key: &str) -> anyhow::Result<&Value> {
        self.meta.get(key).with_context(|| format!("missing GGUF key {key}"))
    }

    /// Raw bytes of a tensor. Panics on an unsupported type; `Weights::validate` checks types first.
    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        let ty = GgmlType::from_u32(t.ty).unwrap();
        let n = ty.row_bytes(t.dims[0]) * t.dims[1..].iter().product::<u64>() as usize;
        let (mmap, data_offset) = &self.files[t.file];
        let o = data_offset + t.offset as usize;
        &mmap[o..o + n]
    }
}
