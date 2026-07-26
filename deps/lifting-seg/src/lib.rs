#[pyo3::pymodule]
mod lifting_seg {
    use rayon::prelude::*;
    use pyo3::prelude::*;
    use numpy::prelude::*;

    #[pyfunction]
    #[pyo3(signature = (rays, ray_feats, verts, radius=0.010))]
    pub fn intersection<'py>(
        py: Python<'py>,
        rays: numpy::PyReadonlyArray3<'py, f64>,
        ray_feats: numpy::PyReadonlyArray2<'py, f64>,
        verts: numpy::PyReadonlyArray2<'py, f64>,
        radius: f64,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let rays_view = rays.as_array();
        let ray_feats_view = ray_feats.as_array();
        let verts_view = verts.as_array();

        let num_rays = rays_view.shape()[0];
        let num_verts = verts_view.shape()[0];

        let mut vert_feats = numpy::ndarray::Array2::<f64>::zeros((num_verts, 3));

        py.detach(|| {
            let num_feats = ray_feats_view.shape()[1];

            let rays = (0..num_rays)
                .map(|i| Ray {
                    org: nalgebra::Point3::new(rays_view[[i, 0, 0]], rays_view[[i, 0, 1]], rays_view[[i, 0, 2]]),
                    end: nalgebra::Point3::new(rays_view[[i, 1, 0]], rays_view[[i, 1, 1]], rays_view[[i, 1, 2]]),
                    feats: ray_feats_view.row(i).to_vec(),
                })
                .collect::<Vec<_>>();

            let mut spheres = (0..num_verts)
                .map(|i| Sphere {
                    origin_index: i,
                    node_index: i,
                    vert: nalgebra::Point3::new(verts_view[[i, 0]], verts_view[[i, 1]], verts_view[[i, 2]]),
                    radius,
                    feats: vec![0.0; num_feats],
                    weights: 1e-8,
                })
                .collect::<Vec<_>>();

            intersection_check(&rays, &mut spheres, num_feats);

            for sphere in spheres.iter() {
                for f in 0..num_feats {
                    vert_feats[[sphere.origin_index, f]] = sphere.feats[f];
                }
            }
        });

        Ok(vert_feats.into_pyarray(py))
    }

    #[derive(Debug)]
    struct Ray {
        org: nalgebra::Point3<f64>,
        end: nalgebra::Point3<f64>,
        feats: Vec<f64>,
    }

    #[derive(Debug)]
    struct Sphere {
        origin_index: usize,
        node_index: usize,
        vert: nalgebra::Point3<f64>,
        radius: f64,
        feats: Vec<f64>,
        weights: f64,
    }

    impl bvh::aabb::Bounded<f64, 3> for Sphere {
        fn aabb(&self) -> bvh::aabb::Aabb<f64, 3> {
            let radius = nalgebra::Vector3::new(self.radius, self.radius, self.radius);
            let min = self.vert - radius;
            let max = self.vert + radius;
            bvh::aabb::Aabb::with_bounds(min, max)
        }
    }

    impl bvh::bounding_hierarchy::BHShape<f64, 3> for Sphere {
        fn set_bh_node_index(&mut self, node_index: usize) {
            self.node_index = node_index
        }

        fn bh_node_index(&self) -> usize {
            self.node_index
        }
    }

    impl Sphere {
        pub fn intersection_with_ray(&self, ray: &bvh::ray::Ray<f64, 3>) -> f64 {
            let oc = ray.origin - self.vert;
            let a = ray.direction.dot(&ray.direction);
            let b = 2.0 * oc.dot(&ray.direction);
            let c = oc.dot(&oc) - self.radius * self.radius;
            let discriminant = b * b - 4.0 * a * c;

            if discriminant < 0.0 {
                return f64::INFINITY;
            }

            let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
            let t2 = (-b + discriminant.sqrt()) / (2.0 * a);

            let t = if t1 >= 0.0 { t1 } else { t2 };

            if t < 0.0 {
                return f64::INFINITY;
            }

            t
        }
    }

    fn intersection_check(rays: &[Ray], spheres: &mut [Sphere], num_feat: usize) {
        println!("{} node BVH tree building...", spheres.len());
        let bvh = <bvh::bvh::Bvh<f64, 3> as bvh::bounding_hierarchy::BoundingHierarchy<f64, 3>>::build_par(spheres);

        println!("Traverse ray casting...");
        let hits = rays
            .par_iter()
            .enumerate()
            .filter_map(|(ray_index, ray_data)| {
                let ray = bvh::ray::Ray::new(ray_data.org, (ray_data.end - ray_data.org).normalize());
                bvh.traverse(&ray, spheres)
                    .into_iter()
                    .map(|sphere| (sphere.origin_index, sphere.intersection_with_ray(&ray)))
                    .filter(|(_, dist)| dist.is_finite())
                    .min_by(|(_, dist0), (_, dist1)| f64::partial_cmp(dist0, dist1).unwrap())
                    .map(|(sphere_index, _)| (ray_index, sphere_index))
            })
            .collect::<Vec<_>>();

        println!("{} ray hits...", hits.len());
        hits.into_iter()
            .for_each(|(ray_index, sphere_index)| {
                let ray = &rays[ray_index];
                let sphere = &mut spheres[sphere_index];
                (0..num_feat).for_each(|i| sphere.feats[i] += ray.feats[i]);
                sphere.weights += 1.0;
            });

        println!("Normalize results...");
        spheres.par_iter_mut()
            .for_each(|sphere| {
                (0..num_feat).for_each(|i| sphere.feats[i] /= sphere.weights);
            });
    }
}
