use std::env;
use std::fs;
extern crate rgsl;

use std::rc::Rc;
use std::cell::RefCell;
use plotters::prelude::*;
use std::time::Instant;
use std::path::Path;
use std::io::{Error, ErrorKind};
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use image::{imageops::FilterType, ImageFormat};

#[macro_use]
extern crate lazy_static;
lazy_static!{
    static ref AIR_REFRACTIVE_INDEX:f64 = 1f64;
    static ref CORNEA_REFRACTIVE_INDEX:f64 = 1.37;
}
macro_rules! clone {
    (@param _) => ( _ );
    (@param $x:ident) => ( mut $x );
    ($($n:ident),+ => move || $body:expr) => (
        {
            $( let $n = $n.clone(); )+
            move || $body
        }
    );
    ($($n:ident),+ => move |$($p:tt),+| $body:expr) => (
        {
            $( let $n = $n.clone(); )+
            move |$(clone!(@param $p),)+| $body
        }
    );
}

fn exp_f(x: &rgsl::VectorF64, f: &mut rgsl::VectorF64, data: &Data) -> rgsl::Value {
    let a = x.get(0);
    let b = x.get(1);
    let c = x.get(2);

    for (i, (x, y)) in data.x.iter().zip(data.y.iter()).enumerate(){
        /* Model Yi = a * x^2 + b * x + c*/
        let yi = a * x.powi(2) + b * x + c; 

        f.set(i, yi - y);

    }
    rgsl::Value::Success
}

fn exp_df(x: &rgsl::VectorF64, J: &mut rgsl::MatrixF64, data: &Data) -> rgsl::Value {

    for (i, x) in data.x.iter().enumerate(){
        /* Jacobian matrix J(i,j) = dfi / dxj, */
        /* where fi = (Yi - yi)/sigma[i],      */
        /*       Yi = A * exp(-lambda * i) + b  */
        /* and the xj are the parameters (A,lambda,b) */
        J.set(i, 0, x.powi(2));
        J.set(i, 1, *x);
        J.set(i, 2, 1f64);
    }
    rgsl::Value::Success
}

fn print_state(iter: usize, s: &rgsl::MultiFitFdfSolver) {
    println!("iter: {} x = {} {} {} |f(x)| = {}", iter,
             s.x().get(0), s.x().get(1), s.x().get(2), rgsl::blas::level1::dnrm2(&s.f()));
}

#[derive(Debug)]
pub struct Data{
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    n: usize
}
fn read_file_into_lines(filename: &str)-> Vec<String>{
    let contents = fs::read_to_string(
        filename).expect(&format!("can't read file {}", filename));
    let rows: Vec<String> = contents.split('\n').map(|s| s.to_string()).collect();
    rows
}

fn bound_in_axis(x: f32, y: f32) -> bool{
    let x_low = -8;
    let x_high = 8;
    let y_low = 0;
    let y_high = 16;
    x > x_low as f32 && x < x_high as f32 && y > y_low as f32 && y < y_high as f32
}
fn plot_parabola(data: &Data, params: &rgsl::VectorF64, additional_points: Vec<Vec<f64>>, fig_path: &String, image_path: &String){
    let root = BitMapBackend::new(fig_path, (1024, 1024)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Optical Distortion Correction", ("sans-serif", 10).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_ranged(8f32..-8f32, 16f32..0f32).unwrap();

    //chart.configure_mesh().draw().unwrap();
    // Plot background image
    //let (w, h) = chart.plotting_area().dim_in_pixel();
    //println!("plotting area: {}, {}", w, h);
    //let image = image::load(
        //BufReader::new(File::open(image_path).unwrap()),
        //ImageFormat::Bmp,
    //).unwrap()
    //.resize_exact(16, 16, FilterType::Nearest);

    //let elem: BitMapElement<_> = ((-8.0, 8.0), image).into();
    //chart.draw_series(std::iter::once(elem)).unwrap();

    // Draw Overlay points
    let x_low = -8;
    let x_high = 8;
    let y_low = 0;
    let y_high = 16;
    let a = params.get(0);
    let b = params.get(1);
    let c = params.get(2);
    // Draw parabola
    chart.draw_series(LineSeries::new(
            (x_low..=x_high).map(|x| {x as f32}).map(|x| (x, x.powi(2) * a as f32 + x * b as f32 + c as f32)),
            &RED,
        )).unwrap()
        .label(format!("y = {:.3}x^2 + {:.3}x + {:.3}", a, b, c)).legend(|(x, y)| PathElement::new(vec![(x, y), (x + 2, y)], &RED));

    // Draw cornea points
    chart.draw_series(PointSeries::of_element(data.x.iter().zip(data.y.iter()).map(|(x, y)| (*x as f32, *y as f32)), 2, ShapeStyle::from(&RED).filled(), 
        &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0, 0), size, style)
        })).unwrap();

    // Plot pupil margins
    chart.draw_series(PointSeries::of_element(additional_points.iter().map(|v| (v[0] as f32, v[1] as f32)), 3, ShapeStyle::from(&BLUE).filled(),
        &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0, 0), size, style)
        })).unwrap();
    
    // Draw inbound ray
    let pm0_x = additional_points[0][0];
    let pm0_boundary = vec![pm0_x, a * pm0_x.powi(2) + b * pm0_x + c];
    let pm1_x = additional_points[1][0];
    let pm1_boundary = vec![pm1_x, a * pm1_x.powi(2) + b * pm1_x + c];
    chart.draw_series(LineSeries::new(
        (y_low..=y_high+8).map(|y| {y as f32}).map(|y| (pm0_x as f32, y)),
        &BLUE)).unwrap();

    chart.draw_series(LineSeries::new(
        (y_low..=y_high+8).map(|y| {y as f32}).map(|y| (pm1_x as f32, y)),
        &BLUE)).unwrap();

    // Draw tangent line
    let tangent_line_0_k = 2f64 * a * pm0_x + b;
    let tangent_line_0_b =  pm0_boundary[1] - tangent_line_0_k * pm0_x;
    let mut tangent_line_low = ((pm0_x - 2f64) * 10f64) as i32;
    let mut tangent_line_high = ((pm0_x + 2f64) * 10f64) as i32;
    //println!("{}, {}", tangent_line_low, tangent_line_high);
    chart.draw_series(LineSeries::new(
        (tangent_line_low..=tangent_line_high).map(|x| x as f32 / 10f32).map(|x| (x, tangent_line_0_k as f32 * x + tangent_line_0_b as f32))
        .filter(
            |(x, y)| bound_in_axis(*x, *y))
            ,&BLACK
    )).unwrap();
    let tangent_line_0_k_vert = -1f64 / tangent_line_0_k;
    let tangent_line_0_b_vert =  pm0_boundary[1] - (-1f64 / tangent_line_0_k) * pm0_x;
    chart.draw_series(LineSeries::new(
        (tangent_line_low..=tangent_line_high).map(|x| x as f32 / 10f32).map(|x| (x, tangent_line_0_k_vert as f32 * x + tangent_line_0_b_vert as f32))
        .filter(
            |(x, y)| bound_in_axis(*x, *y))
            ,&BLACK
    )).unwrap();
    
    // Draw tangent line
    let tangent_line_1_k = 2f64 * a * pm1_x + b;
    let tangent_line_1_b =  pm1_boundary[1] - tangent_line_1_k * pm1_x;
    tangent_line_low = ((pm1_x - 2f64) * 10f64) as i32;
    tangent_line_high = ((pm1_x + 2f64) * 10f64) as i32;
    //println!("{}, {}", tangent_line_low, tangent_line_high);
    chart.draw_series(LineSeries::new(
        (tangent_line_low..=tangent_line_high).map(|x| x as f32 / 10f32).map(|x| (x, tangent_line_1_k as f32 * x + tangent_line_1_b as f32))
            .filter(
            |(x, y)| bound_in_axis(*x, *y))
            ,&BLACK
    )).unwrap();
    let tangent_line_1_k_vert = -1f64 / tangent_line_1_k;
    let tangent_line_1_b_vert =  pm1_boundary[1] - (-1f64 / tangent_line_1_k) * pm1_x;
    chart.draw_series(LineSeries::new(
        (tangent_line_low..=tangent_line_high).map(|x| x as f32 / 10f32).map(|x| (x, tangent_line_1_k_vert as f32 * x + tangent_line_1_b_vert as f32))
        .filter(
            |(x, y)| bound_in_axis(*x, *y))
            ,&BLACK
    )).unwrap();
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw().unwrap();
}

fn curve_fitting(data: Rc<Data>) -> Result<rgsl::MultiFitFdfSolver, Error>{
    //Initialize Parameters
    let num_params = 3;
    let num_instances = data.n;
    let mut status = rgsl::Value::Success;

    let mut J = match rgsl::MatrixF64::new(num_params, num_params){
        Some(mat) => mat,
        None => return Err(Error::new(ErrorKind::Other, "Can't create Jacobian matrix"))
    };
    let mut params_init = [0f64, -1f64, 100f64];
    let mut params = rgsl::VectorView::from_array(&mut params_init);

    rgsl::RngType::env_setup();
    let rng_type = rgsl::rng::default();
    let mut rng = match rgsl::Rng::new(&rng_type){
        Some(r) => r,
        None => return Err(Error::new(ErrorKind::Other, "Can't create rng"))
    };

    let solver_type = rgsl::MultiFitFdfSolverType::lmsder();

    let mut func = rgsl::MultiFitFunctionFdf::new(num_instances, num_params);
    let expb_f = clone!(data => move |x, f| {
        exp_f(&x, &mut f, &*data)
    });
    func.f = Some(Box::new(expb_f));
    let expb_df = clone!(data => move |x, J| {
        exp_df(&x, &mut J, &*data)
    });
    func.df = Some(Box::new(expb_df));
    let expb_fdf = clone!(data => move |x, f, J| {
        exp_f(&x, &mut f, &*data);
        exp_df(&x, &mut J, &*data);

        rgsl::Value::Success
    });
    func.fdf = Some(Box::new(expb_fdf));

    // Create a solver
    let mut solver = match rgsl::MultiFitFdfSolver::new(&solver_type, num_instances, num_params){
        Some(s) => s,
        None => return Err(Error::new(ErrorKind::Other, "can't create solver"))
    };
    solver.set(&mut func, &params.vector());
    let mut iter = 0;
    loop {
        iter += 1;
        status = solver.iterate();

        println!("status = {}", rgsl::error::str_error(status));
        print_state(iter, &solver);

        if status != rgsl::Value::Success {
            //return Err(Error::new(ErrorKind::TimedOut, "Reconstruction failed"));
            break;
        }
        //println!("dx: {:?}", &solver.dx());
        //println!("position: {:?}", &s.position());
        status = rgsl::multifit::test_delta(&solver.dx(), &solver.x(), 1e-4, 1e-4);
        if status != rgsl::Value::Continue || iter >= 500 {
            break;
        }
    }
    println!("Done");
    println!("params: {:?}", &solver.x());
    Ok(solver)
}

fn apply_optical_distortion_correction(params: &rgsl::VectorF64, points: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    
    let a = params.get(0);
    let b = params.get(1);
    let c = params.get(2);
    let mut new_points: Vec<Vec<f64>> = Vec::new();
    for point in points.iter(){
        let k1 = 2f64 * a * point[0] + b;
        
        let theta_1 = k1.atan();
        let mut theta_2 = 0f64;
        let mut theta = 0f64;
        let mut new_point: Vec<f64> = Vec::new();
        if theta_1 > 0f64{
            theta_2 = theta_1 * *AIR_REFRACTIVE_INDEX/ *CORNEA_REFRACTIVE_INDEX;
            theta = theta_1 - theta_2;
        }else{
            theta_2 = theta_1.abs() / *CORNEA_REFRACTIVE_INDEX;
            theta = -1f64 * (theta_1.abs() - theta_2);
        }
        println!("theta: {}", theta);
        let boundary_point = vec![point[0], a * point[0].powi(2) + b * point[0] + c];
        let new_length = (boundary_point[1] - point[1]) / *CORNEA_REFRACTIVE_INDEX;
        new_point.push(boundary_point[0] + new_length * theta.sin());
        new_point.push(boundary_point[1] - new_length * theta.cos());
        println!("old: {:?}, new: {:?}", point, new_point);
        new_points.push(new_point);
    }
    new_points
}
/*
fn run_on_folder(folder: &str){
    let mut total_time_ms = 0;
    let mut iter = 0;
    for entry in fs::read_dir(folder).unwrap(){
        let entry = entry.unwrap();
        let file_path = entry.path();
        if !file_path.is_dir(){
            let lines = read_file_into_lines(file_path.to_str().unwrap());
            println!("process: {:?}", file_path);
            // Create Data
            let mut data = Rc::new(RefCell::new(Data{
                x: Vec::<f64>::new(),
                y: Vec::<f64>::new(),
                n: 0
            }));

            for (i, line) in lines.iter().enumerate(){
                let mut data = *data.borrow_mut();
                let numbers: Vec<f64> = line.split(" ").filter_map(|v| v.parse::<f64>().ok()).collect();
                //println!("data: {:?}", numbers);
                if numbers.len() > 1{
                    data.x.push(*numbers.get(0).unwrap());
                    data.y.push(*numbers.get(1).unwrap());
                    data.n += 1;
                }
            }

            //println!("Data: {:?}", data);
    
            if data.borrow().n >= 3{
                let fig_path: String = format!("./figs/{}.png", file_path.file_name().unwrap().to_str().unwrap());
                let now = Instant::now();
                let curve_params = curve_fitting(data); 
                total_time_ms += now.elapsed().as_micros();
                iter += 1;
            }
        }
    }
    println!("run {} iterations", iter);
    println!("average time: {} microsecond", total_time_ms/ iter);
}
*/
fn main() {
    let args: Vec<String> = env::args().collect();
    let path = Path::new(&args[1]);
    if path.is_dir(){
       //run_on_folder(&path.to_str().unwrap()); 
       println!("TODO: fix bug in run folder");
    }else{
        let lines = read_file_into_lines(path.to_str().unwrap());
        let mut total_time_micros = 0;
        let mut iter = 0;

        let mut file = File::create("cross_corrected.keypoint").unwrap();
        file.write_all(b"scanId\teyeId\tbscanId\tpm0xmm\tpm0ymm\tpm0zmm\tpm1xmm\tpm1ymm\tpm1zmm\n");
        for line in lines.iter().skip(1){
            let elements: Vec<&str> = line.split(",").collect();
            if elements.len() > 8{
                let pm0: Vec<f64> = vec![elements[3], elements[4]].iter().filter_map(|e| e.parse::<f64>().ok()).collect();
                let pm1: Vec<f64> = vec![elements[5], elements[6]].iter().filter_map(|e| e.parse::<f64>().ok()).collect();
                let cp_x: Vec<f64> = elements[7].split(" ").filter_map(|v| v.parse::<f64>().ok()).collect();
                let cp_y: Vec<f64> = elements[8].split(" ").filter_map(|v| v.parse::<f64>().ok()).collect();
                let num_kp = cp_y.len();
                let mut data = Rc::new(Data{
                    x: cp_x,
                    y: cp_y,
                    n: num_kp
                });
                //println!("{:?}", data.x);
                if data.n >= 3{
                    let now = Instant::now();
                    let solver = match curve_fitting(Rc::clone(&data)){
                        Ok(p) => p,
                        Err(e) => panic!("can't reconstruct curve")
                    }; 
                    println!("{:?}", solver.x());
                    let mut add_points = vec![pm0, pm1];
                    let corrected_points = apply_optical_distortion_correction(&solver.x(), &add_points);
                    if elements[2] == "x"{
                        let output_str = format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", elements[0], 
                        elements[1], elements[2], 
                        corrected_points[0][0], 0, corrected_points[0][1],
                        corrected_points[1][0], 0, corrected_points[1][1]);
                        file.write_all(output_str.as_bytes()).unwrap();
                    }else{
                        let output_str = format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", elements[0], 
                        elements[1], elements[2], 
                        -1.53, corrected_points[0][0], corrected_points[0][1],
                        -1.53, corrected_points[1][0], corrected_points[1][1]);
                        file.write_all(output_str.as_bytes()).unwrap();
                    }
                    //println!("{:?}", corrected_points);
                    total_time_micros += now.elapsed().as_micros();
                    iter += 1;
                    add_points.extend(corrected_points);
                    let image_path = format!("./resource/TestMotility/OD/{}/0{}.bmp", elements[2], elements[0]);
                    println!("image: {}", image_path);
                    let fig_path: String = format!("./figs_mm/{}_{}.png", elements[0], elements[2]);
                    //plot_parabola(&data, &solver.x(), add_points, &fig_path, &image_path);
                }
            }else{
                println!("total elements: {}", elements.len());
                println!("Can't process {}", line);
            }
        }
        println!("Total iteration: {}", iter);
        println!("Average time: {}", total_time_micros / iter);
    } 
}
