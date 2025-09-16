use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

const NUM_DIRS: usize = 4; // N, S, E, W in this fixed order

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir { North = 0, South = 1, East = 2, West = 3 }

impl Dir {
    #[inline]
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "north" => Some(Dir::North),
            "south" => Some(Dir::South),
            "east"  => Some(Dir::East),
            "west"  => Some(Dir::West),
            _ => None,
        }
    }

    #[allow(dead_code)]
    #[inline]
    fn opposite(self) -> Self {
        match self {
            Dir::North => Dir::South,
            Dir::South => Dir::North,
            Dir::East  => Dir::West,
            Dir::West  => Dir::East,
        }
    }
}

#[derive(Debug, Clone)]
struct Graph {
    // id -> name
    names: Vec<String>,
    // alive flag per colony (future steps will flip to false on destruction)
    alive: Vec<bool>,
    // compact neighbors: only live edges, kept in sync on sever
    neighbors: Vec<[u32; NUM_DIRS]>,
    neighbor_len: Vec<u8>,
}

impl Graph {
    fn new<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let f = File::open(path)?;
        let r = BufReader::new(f);

        // First pass: assign ids to all names and collect raw neighbor strings per line.
        let mut id_by_name: HashMap<String, u32> = HashMap::new();
        let mut lines_raw: Vec<(u32, Vec<(Dir, String)>)> = Vec::new();

        for line in r.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() { continue; }

            let mut parts = line.split_whitespace();
            let name = parts.next().ok_or_else(|| anyhow::anyhow!("malformed line: missing name"))?;

            let id = if let Some(&existing_id) = id_by_name.get(name) {
                existing_id
            } else {
                let new_id = id_by_name.len() as u32;
                id_by_name.insert(name.to_string(), new_id);
                new_id
            };
            let mut neighs: Vec<(Dir, String)> = Vec::with_capacity(4);

            for tok in parts {
                let mut kv = tok.split('=');
                let k = kv.next().ok_or_else(|| anyhow::anyhow!("malformed token: {}", tok))?;
                let v = kv.next().ok_or_else(|| anyhow::anyhow!("malformed token, missing '=': {}", tok))?;
                if kv.next().is_some() { anyhow::bail!("extra '=' in token: {}", tok); }
                let dir = Dir::from_str(k).ok_or_else(|| anyhow::anyhow!("unknown direction: {}", k))?;
                neighs.push((dir, v.to_string()));
            }

            lines_raw.push((id, neighs));
        }

        // Allocate graph with known size.
        let n = id_by_name.len();
        let mut names = vec![String::new(); n];
        for (name, &id) in &id_by_name {
            names[id as usize] = name.clone();
        }
        let mut adj = vec![[None; NUM_DIRS]; n];
        let alive = vec![true; n];

        // Second pass: wire adjacency.
        for (id, neighs) in lines_raw.into_iter() {
            for (dir, vname) in neighs.into_iter() {
                let &nid = id_by_name.get(&vname)
                    .ok_or_else(|| anyhow::anyhow!("neighbor not defined in file: {}", vname))?;
                adj[id as usize][dir as usize] = Some(nid);
            }
        }

        // Build compact neighbor arrays from adj
        let mut neighbors = vec![[0u32; NUM_DIRS]; n];
        let mut neighbor_len = vec![0u8; n];
        for (i, row) in adj.iter().enumerate() {
            let mut k: usize = 0;
            if let Some(n0) = row[0] { neighbors[i][k] = n0; k += 1; }
            if let Some(n1) = row[1] { neighbors[i][k] = n1; k += 1; }
            if let Some(n2) = row[2] { neighbors[i][k] = n2; k += 1; }
            if let Some(n3) = row[3] { neighbors[i][k] = n3; k += 1; }
            neighbor_len[i] = k as u8;
        }

        Ok(Self { names, alive, neighbors, neighbor_len })
    }
}

// A tiny, fast per-ant RNG (xorshift64*)
#[derive(Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    #[inline(always)] fn new(seed: u64) -> Self { Self { state: seed.max(1) } }
    #[inline(always)] fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    #[inline(always)] fn next_u32(&mut self) -> u32 { (self.next_u64() >> 32) as u32 }
    #[inline(always)] fn gen_index(&mut self, m: u32) -> u32 { // returns in [0, m)
        debug_assert!(m > 0);
        // Fast modulo; m <= 4 here so bias is irrelevant.
        self.next_u32() % m
    }
}

struct AntSim {
    g: Graph,
    // ant i -> colony id
    positions: Vec<u32>,
    live_ant_ids: Vec<usize>,
    ant_index_in_live: Vec<usize>,
    rngs: Vec<XorShift64>,

    // per-iteration scratch
    next_pos: Vec<u32>,
    in_count: Vec<u32>,
    fight_first: Vec<u32>,   // first incoming ant id per colony (u32::MAX = none)
    fight_second: Vec<u32>,  // second incoming ant id per colony (u32::MAX = none)
    touched_cols: Vec<u32>,    // which cols had in_count incremented this tick
}

impl AntSim {
    fn new(g: Graph, n_ants: usize, seed: u64) -> Self {
        let ncols = g.names.len();
        assert!(ncols > 0, "empty map");
        let mut rng = XorShift64::new(seed);
        let mut positions = Vec::with_capacity(n_ants);
        let mut rngs = Vec::with_capacity(n_ants);
        for i in 0..n_ants {
            // place randomly among alive colonies
            let cid = rng.next_u32() as usize % ncols;
            positions.push(cid as u32);
            // seed per-ant rng differently (mix with i)
            rngs.push(XorShift64::new(seed ^ ((i as u64).wrapping_mul(0x9E3779B97F4A7C15))));
        }
        let live_ant_ids = (0..n_ants).collect::<Vec<_>>();
        let ant_index_in_live = (0..n_ants).collect::<Vec<_>>();

        let invalid = u32::MAX;

        Self {
            g,
            positions,
            live_ant_ids,
            ant_index_in_live,
            rngs,
            next_pos: vec![0; n_ants],
            in_count: vec![0; ncols],
            fight_first: vec![invalid; ncols],
            fight_second: vec![invalid; ncols],
            touched_cols: Vec::with_capacity(n_ants.min(ncols)),
        }
    }

    #[inline]
    fn remove_live_ant(&mut self, ant_id: usize) {
        let idx = self.ant_index_in_live[ant_id];
        if idx == usize::MAX { return; }
        let last = self.live_ant_ids.pop().unwrap();
        if last != ant_id {
            self.live_ant_ids[idx] = last;
            self.ant_index_in_live[last] = idx;
        }
        self.ant_index_in_live[ant_id] = usize::MAX;
    }

    #[inline(always)]
    fn decide_moves(&mut self) {
        // Sparse clear for touched colonies
        let invalid = u32::MAX;
        for &c in &self.touched_cols {
            // reset only those we touched last tick
            self.in_count[c as usize] = 0;
            self.fight_first[c as usize] = invalid;
            self.fight_second[c as usize] = invalid;
        }
        self.touched_cols.clear();
    
        let mut idx = 0usize;
        while idx < self.live_ant_ids.len() {
            let i = self.live_ant_ids[idx];
            let cur = self.positions[i] as usize;
            if !self.g.alive[cur] {
                self.next_pos[i] = cur as u32;
                idx += 1;
                continue;
            }
    
            // Use compact neighbor list
            let len = self.g.neighbor_len[cur] as usize;
            let dst = if len == 0 {
                // trapped: ant dies immediately
                self.remove_live_ant(i);
                // do not advance idx since we swapped another ant into this idx
                continue;
            } else {
                let k = self.rngs[i].gen_index(len as u32) as usize;
                unsafe { *self.g.neighbors.get_unchecked(cur).get_unchecked(k) }
            };
    
            self.next_pos[i] = dst;
    
            // Increment in_count[dst] and record first two ant IDs
            let entry = &mut self.in_count[dst as usize];
            match *entry {
                0 => {
                    self.touched_cols.push(dst);
                    self.fight_first[dst as usize] = i as u32;
                }
                1 => {
                    self.fight_second[dst as usize] = i as u32;
                }
                _ => { /* ignore beyond two; we only need a pair to print */ }
            }
            *entry += 1;

            idx += 1;
        }
    }

    #[inline(always)]
    fn sever_colony(&mut self, col: u32) {
        let c = col as usize;
        self.g.alive[c] = false;
        // Remove all neighbors of c in compact list and back-edges
        let len_c = self.g.neighbor_len[c] as usize;
        for k in 0..len_c {
            let n = self.g.neighbors[c][k] as usize;
            // remove back-edge in neighbor compact list
            let len_n = self.g.neighbor_len[n] as usize;
            for t in 0..len_n {
                if self.g.neighbors[n][t] as usize == c {
                    // swap_remove within fixed array region [0..len)
                    let last_idx = len_n - 1;
                    self.g.neighbors[n][t] = self.g.neighbors[n][last_idx];
                    self.g.neighbor_len[n] = (last_idx) as u8;
                    break;
                }
            }
        }
        // clear c's own list
        self.g.neighbor_len[c] = 0;
    }

    fn resolve_fights_and_destroy<W: std::fmt::Write>(&mut self, out: &mut W) {
        let invalid = u32::MAX;
        
        // First pass: collect colonies to destroy and print messages
        let mut to_destroy = Vec::new();
        for &col in &self.touched_cols {
            let c = col as usize;
            if self.in_count[c] >= 2 {
                // choose two fighter IDs deterministically
                let a = self.fight_first[c];
                let mut b = self.fight_second[c];
                if b == invalid { b = a; } // extremely unlikely, but keep safe
                let name = &self.g.names[c];

                // print message
                let _ = writeln!(out, "{name} has been destroyed by ant {a} and ant {b}!");
                
                to_destroy.push(col);
            }
        }
        
        // Second pass: actually destroy the colonies
        for col in to_destroy {
            self.sever_colony(col);
        }
    }

    #[inline(always)]
    fn commit_moves(&mut self) {
        let mut idx = 0usize;
        while idx < self.live_ant_ids.len() {
            let i = self.live_ant_ids[idx];
            let dst = self.next_pos[i] as usize;
            if self.g.alive[dst] {
                self.positions[i] = dst as u32;
                idx += 1;
            } else {
                // destination blown up this tick -> ant dies
                self.remove_live_ant(i);
                // do not advance idx; swapped ant now at this index
            }
        }
    }

    fn step<W: std::fmt::Write>(&mut self, out: &mut W) {
        self.decide_moves();
        self.resolve_fights_and_destroy(out);
        self.commit_moves();
    }

    #[inline]
    fn live_ants(&self) -> usize { self.live_ant_ids.len() }

    /// Run until all ants dead or `max_iters` reached. Fight logs appended to `out`.
    fn run(&mut self, max_iters: u32, out: &mut String) {
        for _ in 0..max_iters {
            if self.live_ants() == 0 { break; }
            self.step(out);
        }
    }    
}

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        eprintln!("Usage:\n  ant_mania <map_path> <num_ants> [seed]\n  ant_mania bench <map_path> [trials] [seed] <count1> <count2> ...");
        std::process::exit(2);
    }

    if args[0] == "bench" {
        // Usage: bench <map_path> <num_ants> <trials> [seed]
        if args.len() < 4 {
            eprintln!("Usage: ant_mania bench <map_path> <num_ants> <trials> [seed]");
            std::process::exit(2);
        }
        let map_path = &args[1];
        let n_ants: usize = args[2].parse()?;
        let trials: u32 = args[3].parse()?;
        let seed: u64 = if args.len() >= 5 { args[4].parse()? } else { 0xbadc0de_u64 };

        println!("bench map={} ants={} trials={} seed={}", map_path, n_ants, trials, seed);
        let mut total_ns: u128 = 0;
        for t in 0..trials {
            let t0 = Instant::now();
            let g = Graph::new(map_path)?;
            let mut sim = AntSim::new(g, n_ants, seed ^ (t as u64));
            let mut fight_log = String::new();
            sim.run(10_000, &mut fight_log);
            let dt = t0.elapsed();
            total_ns += dt.as_nanos();
        }
        let avg_ns = total_ns / (trials as u128);
        let avg_dur = std::time::Duration::from_nanos(avg_ns as u64);
        eprintln!("Average time: {:?}", avg_dur);
        return Ok(());
    }

    if args.len() < 2 {
        eprintln!("Usage: ant_mania <map_path> <num_ants> [seed]");
        std::process::exit(2);
    }
    let n_ants: usize = args[1].parse()?;
    let seed: u64 = if args.len() >= 3 { args[2].parse()? } else { 0xbadc0de_u64 };
    
    let map_path = &args[0];
    
    // timer starts before graph construction
    let sim_t0 = Instant::now();
    let g = Graph::new(map_path)?;
    let mut sim = AntSim::new(g, n_ants, seed);

    let mut fight_log = String::new();
    sim.run(10_000, &mut fight_log);
    let sim_elapsed = sim_t0.elapsed();

    // Print all fight messages first
    print!("{fight_log}");
    // Summary stats
    let total_cols = sim.g.names.len();
    let alive_cols = sim.g.alive.iter().filter(|&&a| a).count();
    let destroyed_cols = total_cols - alive_cols;
    let ants_remaining = sim.live_ants();
    eprintln!("Simulation time: {:?}", sim_elapsed);
    eprintln!("Destroyed colonies: {}", destroyed_cols);
    eprintln!("Ants remaining: {}", ants_remaining);

    Ok(())
}