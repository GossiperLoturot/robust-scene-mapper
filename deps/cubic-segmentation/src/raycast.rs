use color_print::*;
use indicatif::*;
use rayon::prelude::*;

use crate::triangle;

#[derive(Debug)]
pub struct Frame {
    pub extrinsics: nalgebra::Matrix4<f32>,
    pub intrinsics: nalgebra::Matrix3<f32>,
    pub size: nalgebra::Vector2<u32>,
}

#[derive(Debug)]
pub struct Segmentation {
    pub frame_id: u32,
    pub class_name: String,
    pub confidence: f32,
    pub mask: Vec<u8>,
}

#[derive(Debug)]
pub struct DenseSegmentation {
    pub extrinsics: nalgebra::Matrix4<f32>,
    pub intrinsics: nalgebra::Matrix3<f32>,
    pub size: nalgebra::Vector2<u32>,
    pub mask: Vec<f32>,
}

#[derive(Debug)]
pub struct Ray {
    pub origin: nalgebra::Point3<f32>,
    pub direction: nalgebra::Vector3<f32>,
    pub class_vec: Vec<f32>,
}

pub fn create_rays(conn: &rusqlite::Connection, n_class: usize, class_names: &[&str]) -> Vec<Ray> {
    // 一枚の画像ごとにカメラパラメータと画像サイズの組み合わせ（フレーム）を抽出
    cprintln!("<g!><s>Load Frames</></>");
    let mut stmt = conn
        .prepare("SELECT extrinsics, intrinsics, width, height FROM images ORDER BY id")
        .unwrap();
    let frames = stmt
        .query_map([], |row| {
            // read extrinsics matrix
            let blob = row.get::<_, String>(0)?;
            let data = serde_json::from_str::<Vec<f32>>(&blob).unwrap();
            let extrinsics = nalgebra::Matrix4::from_row_slice(&data);
            // read intrinsics matrix
            let blob = row.get::<_, String>(1)?;
            let data = serde_json::from_str::<Vec<f32>>(&blob).unwrap();
            let intrinsics = nalgebra::Matrix3::from_row_slice(&data);
            // read frame image size
            let width = row.get::<_, u32>(2)?;
            let height = row.get::<_, u32>(3)?;
            let size = nalgebra::Vector2::new(width, height);
            // build camera
            Ok(Frame {
                extrinsics,
                intrinsics,
                size,
            })
        })
        .unwrap()
        .map(|result| result.unwrap())
        .collect::<Vec<_>>();
    println!("{} frames.", frames.len());

    // セグメンテーションのデータを読み込み
    cprintln!("<g!><s>Load Segmentations</></>");
    let mut stmt = conn
        .prepare("SELECT image_id, class_name, confidence, mask FROM segmentations ORDER BY id")
        .unwrap();
    // read segmentations data
    let segmentations = stmt
        .query_map([], |row| {
            let frame_id = row.get::<_, u32>(0)?;
            let class_name = row.get::<_, String>(1)?;
            let confidence = row.get::<_, f32>(2)?;
            let blob = row.get::<_, Vec<u8>>(3)?;
            Ok((frame_id, class_name, confidence, blob))
        })
        .unwrap()
        .map(|result| result.unwrap())
        .collect::<Vec<_>>();
    // build segmentation structs
    let segmentations = segmentations
        .into_par_iter()
        .progress()
        .map(|(frame_id, class_name, confidence, blob)| {
            let frame = image::load_from_memory(&blob).unwrap().to_luma8();
            let mask = frame.into_raw();
            Segmentation {
                frame_id,
                class_name,
                confidence,
                mask,
            }
        })
        .collect::<Vec<_>>();
    println!("{} segmentations.", segmentations.len());

    // フレームとセグメンテーションのデータを結合（1:Nの左側結合）
    cprintln!("<g!><s>Build Dense Segmentation</></>");
    // initialize dense segmentation structs
    let mut denses = frames
        .into_par_iter()
        .progress()
        .map(|frame| {
            let width = frame.size.x as usize;
            let height = frame.size.y as usize;
            let mask = vec![0f32; n_class * width * height];
            DenseSegmentation {
                extrinsics: frame.extrinsics,
                intrinsics: frame.intrinsics,
                size: frame.size,
                mask,
            }
        })
        .collect::<Vec<_>>();
    // populate dense segmentation masks
    segmentations
        .into_iter()
        .progress()
        .filter_map(|segmentation| {
            let class_id = class_names
                .iter()
                .position(|name| *name == segmentation.class_name.as_str())?;
            Some((segmentation, class_id))
        })
        .for_each(|(segmenation, class_id)| {
            let dense = &mut denses[segmenation.frame_id as usize];
            segmenation
                .mask
                .into_iter()
                .enumerate()
                .filter(|(_, v)| *v > 0)
                .for_each(|(i, _)| dense.mask[class_id + n_class * i] = segmenation.confidence)
        });
    println!("{} dense segmenation.", denses.len());

    // フレームとセグメンテーション、デプスの組み合わせから1ピクセルごとにレイを構築
    cprintln!("<g!><s>Build rays</></>");
    let rays = denses
        .par_iter()
        .progress()
        .enumerate()
        .map(|(_, dense)| {
            let e_mat = dense.extrinsics.try_inverse().unwrap();
            let i_mat = dense.intrinsics.try_inverse().unwrap();

            let origin = e_mat.transform_point(&nalgebra::Point3::origin());

            let matrix = (0..dense.size.y).flat_map(|y| (0..dense.size.x).map(move |x| (x, y)));
            matrix
                .map(|(x, y)| {
                    let pixel = nalgebra::Point3::new(x as f32 + 0.5, y as f32 + 0.5, 1.0);
                    let nic = i_mat * pixel;
                    let direction = (e_mat.transform_point(&nic) - origin).normalize();

                    let cursor = n_class * (x + y * dense.size.x) as usize;
                    let class_vec = dense.mask[cursor..cursor + n_class].to_vec();

                    Ray {
                        origin,
                        direction,
                        class_vec,
                    }
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();
    println!("{} rays", rays.len());

    rays
}

pub fn task_raycast(conn: &mut rusqlite::Connection, n_class: usize, class_names: &[&str]) {
    println!("{} class names.", n_class);

    let mut rays = create_rays(conn, n_class, class_names);
    let mut triangles = triangle::create_triangles(conn, n_class);

    cprintln!("<g!><s>Build BVH</></>");
    let bvh =
        <bvh::bvh::Bvh<f32, 3> as bvh::bounding_hierarchy::BoundingHierarchy<f32, 3>>::build_par(
            &mut triangles,
        );
    println!("BVH built.");

    // レイのヒットした三角面を計算
    cprintln!("<g!><s>Ray casting</></>");
    let hits = rays
        .par_iter()
        .progress()
        .enumerate()
        .filter_map(|(ray_index, ray_data)| {
            let ray = bvh::ray::Ray::new(ray_data.origin, ray_data.direction);
            bvh.traverse(&ray, &triangles)
                .into_iter()
                .map(|t| (t.face_index, ray.intersects_triangle(&t.v1, &t.v2, &t.v3)))
                .filter(|(_, x)| x.distance.is_finite())
                .min_by(|(_, x), (_, y)| f32::partial_cmp(&x.distance, &y.distance).unwrap())
                .map(|(triangle_index, _)| (ray_index, triangle_index))
        })
        .collect::<Vec<_>>();
    println!("{} ray hits.", hits.len());
    hits.into_iter()
        .progress()
        .for_each(|(ray_index, triangle_index)| {
            // 三角面ごとに各ラベルの信頼度を集計
            let ray = &mut rays[ray_index];
            let triangle = &mut triangles[triangle_index as usize];
            (0..n_class).for_each(|i| triangle.class_vec[i] += ray.class_vec[i]);
        });

    // 各ラベルの信頼度を正規化
    cprintln!("<g!><s>Normalize class vectors</></>");
    triangles.par_iter_mut().progress().for_each(|triangle| {
        let sum = triangle.class_vec.iter().sum::<f32>();
        if sum > 0.0 {
            let rcp = sum.recip();
            triangle.class_vec.iter_mut().for_each(|v| *v *= rcp);
        }
    });
    println!("Class vectors normalized.");

    // 三角面ごとの各ラベルの正規化された信頼度を保存
    cprintln!("<g!><s>Save results</></>");
    let tx = conn.transaction().unwrap();
    {
        // create triangles table
        tx.execute("DROP TABLE IF EXISTS class_vecs", ()).unwrap();
        tx.execute(
            "CREATE TABLE IF NOT EXISTS class_vecs (
            id INTEGER PRIMARY KEY NOT NULL,
            class_vec TEXT NOT NULL)",
            (),
        )
        .unwrap();
        // insert triangles
        let mut stmt = tx
            .prepare("INSERT INTO class_vecs (id, class_vec) VALUES (?, ?)")
            .unwrap();
        triangles
            .into_iter()
            .progress()
            .enumerate()
            .for_each(|(i, triangle)| {
                let face = i as u32;
                let class_vec = serde_json::to_string(&triangle.class_vec).unwrap();
                stmt.execute((face, class_vec)).unwrap();
            });
    }
    tx.commit().unwrap();
    println!("Results saved.");
}
