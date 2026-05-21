use color_print::*;
use indicatif::*;
use rayon::prelude::*;

#[derive(Debug)]
pub struct Triangle {
    pub node_index: u32,
    pub v1: nalgebra::Point3<f32>,
    pub v2: nalgebra::Point3<f32>,
    pub v3: nalgebra::Point3<f32>,
    pub face_index: u32,
    pub class_vec: Vec<f32>,
}

impl bvh::aabb::Bounded<f32, 3> for Triangle {
    fn aabb(&self) -> bvh::aabb::Aabb<f32, 3> {
        let v1 = self.v1;
        let v2 = self.v2;
        let v3 = self.v3;
        let min = v1.inf(&v2).inf(&v3);
        let max = v1.sup(&v2).sup(&v3);
        bvh::aabb::Aabb::with_bounds(min, max)
    }
}

impl bvh::bounding_hierarchy::BHShape<f32, 3> for Triangle {
    fn set_bh_node_index(&mut self, node_index: usize) {
        self.node_index = node_index as u32
    }

    fn bh_node_index(&self) -> usize {
        self.node_index as usize
    }
}

pub fn create_triangles(conn: &rusqlite::Connection, n_class: usize) -> Vec<Triangle> {
    cprintln!("<g!><s>Load Vertices</></>");
    let mut stmt = conn
        .prepare("SELECT x, y, z FROM verts ORDER BY id")
        .unwrap();
    let verts = stmt
        .query_map([], |row| {
            Ok(nalgebra::Point3::new(
                row.get::<_, f32>(0)?,
                row.get::<_, f32>(1)?,
                row.get::<_, f32>(2)?,
            ))
        })
        .unwrap()
        .map(|result| result.unwrap())
        .collect::<Vec<_>>();
    println!("{} vertices.", verts.len());

    cprintln!("<g!><s>Load Faces</></>");
    let mut stmt = conn
        .prepare("SELECT v1, v2, v3 FROM faces ORDER BY id")
        .unwrap();
    let faces = stmt
        .query_map([], |row| {
            Ok([
                row.get::<_, usize>(0)?,
                row.get::<_, usize>(1)?,
                row.get::<_, usize>(2)?,
            ])
        })
        .unwrap()
        .map(|result| result.unwrap())
        .collect::<Vec<_>>();
    println!("{} faces.", faces.len());

    cprintln!("<g!><s>Build trianlges</></>");
    let triangles = faces
        .into_par_iter()
        .progress()
        .enumerate()
        .map(|(i, face)| Triangle {
            node_index: i as u32,
            face_index: i as u32,
            v1: verts[face[0]],
            v2: verts[face[1]],
            v3: verts[face[2]],
            class_vec: vec![0f32; n_class],
        })
        .collect::<Vec<_>>();
    println!("{} triangles", triangles.len());

    triangles
}
