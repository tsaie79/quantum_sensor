import os

from fireworks import LaunchPad, Workflow
from pymatgen.io.vasp.inputs import Lattice
from qubitPack.tool_box import *
from my_atomate_jyt.vasp.workflows.wf_full import get_wf_full_hse
from my_atomate_jyt.vasp.fireworks.pyzfs import PyzfsFW
from my_atomate_jyt.vasp.powerups import (
    add_additional_fields_to_taskdocs,
    remove_todb,
    bash_scp_files,
    jmodify_to_soc,
    write_PMGObjects
)
from atomate.vasp.powerups import (
    add_tags,
    add_modify_incar,
    add_modify_kpoints,
    add_modify_potcar,
    set_queue_options,
    set_execution_options,
    preserve_fworker,
    clean_up_files
)

from atomate.vasp.config import *
import sys
from JPack.projects.antisiteQubit.wf_defect import ZPLWF
import numpy as np

import glob



def standard_defect(defect_type="substitutions", distort=0.0, category="monolayer_biaxial", dz=0.0,
                    biaxial_ratio=None, fworker="owls"): #1e-4
    wfs = []
    db_name, col_name = "owls", "mx2_antisite_pc"
    # db_name, col_name = "single_photon_emitter", "pc"
    col = get_db(db_name, col_name, port=27017).collection
    mx2s = col.find({"task_id":{"$in": [3097]}})
    # 3091: S-W, 3083: Se-W, 3093: Te-W, 3097:Mo-S, 3094: Mo-Se, 3102:Mo-Te

    for mx2 in mx2s:
        pc = Structure.from_dict(mx2["output"]["structure"])
        print("orig_lattice: {}".format(pc.lattice.matrix))
        if biaxial_ratio:
            new_lattice = Lattice(pc.lattice.matrix.dot(np.eye(3)*(1+biaxial_ratio)))
            pc.lattice = new_lattice
            print("after_biax_lattice: {}".format(pc.lattice.matrix))

        scaling = find_scaling_for_2d_defect(pc, 15)[0]
        # area = scaling[0]*scaling[1]
        area = 25
        geo_spec = {area*pc.num_sites: [20]}

        defect = ChargedDefectsStructures(pc, antisites_flag=True).defects
        cation, anion = find_cation_anion(pc)

        for sub in range(len(defect[defect_type])):
            print(cation, anion)
            # cation vacancy
            #"as_1_{}_on_{}"
            #"vac_1_{}"
            print(defect[defect_type][sub]["name"])
            if "as_1_{}_on_{}".format(cation, anion) not in defect[defect_type][sub]["name"]:
                continue
            for na, thicks in geo_spec.items():
                for thick in thicks:
                    gen_defect = GenDefect(
                        orig_st=pc,
                        defect_type=(defect_type, sub),
                        natom=na,
                        vacuum_thickness=thick,
                        distort=distort,
                        sub_on_side=None,
                    )
                    # special_st = Structure.from_file("/home/tug03990/wse0_456.vasp")
                    # move_site(special_st, [25], dz)
                    wf = get_wf_full_hse(
                        structure=gen_defect.defect_st,
                        task_arg=dict(lcharg=True),
                        charge_states=[0],
                        gamma_only=False,
                        gamma_mesh=True,
                        nupdowns=[2],
                        task="hse_relax-hse_scf",
                        vasptodb={
                            "NN": gen_defect.NN,
                            "defect_entry": gen_defect.defect_entry,
                            "lattice_constant": "HSE",
                            "perturbed": {"biaxial_ratio": 1+biaxial_ratio},
                            "pc_from": "{}/{}/{}".format(db_name, col_name, mx2["task_id"]),
                            "PS": "test",
                        },
                        wf_addition_name="{}:{}".format(na, thick),
                    )
                    wf_name = wf.name
                    fws = wf.fws
                    pyzfsfw = PyzfsFW(parents=fws[-1], structure=gen_defect.defect_st)
                    fws.append(pyzfsfw)
                    wf = Workflow(fws)

                    wf = add_tags(wf,
                                  [
                                      {
                                          "PS": "test",
                                          "NN": gen_defect.NN,
                                          "perturbed": {"biaxial_ratio": 1+biaxial_ratio},
                                          "pc_from": "{}/{}/{}".format(db_name, col_name, mx2["task_id"]),
                                      }
                                  ]
                                  )

                    # wf = bash_scp_files(
                    #     wf,
                    #     dest="/home/jengyuantsai/Research/projects/single_photon_emitter/{}".format(category),
                    #     port=12346,
                    #     fw_name_constraint=wf.fws[-1].name,
                    #     task_name_constraint="VaspToDb"
                    # )

                    # wf = clean_up_files(wf, files=["*"], task_name_constraint="VaspToDb",
                    #                     fw_name_constraint="HSE_scf")

                    # wf = remove_todb(wf, fw_name_constraint=wf.fws[0].name)

                    wf = add_modify_incar(wf, {"incar_update": {"EDIFF": 1e-5, "EDIFFG": -0.02}},
                                          fw_name_constraint=wf.fws[0].name)

                    wf = set_queue_options(wf, "48:00:00", fw_name_constraint="HSE_relax")
                    wf = set_queue_options(wf, "24:00:00", fw_name_constraint="HSE_scf")
                    wf = set_queue_options(wf, "1:30:00", fw_name_constraint=wf.fws[-1].name)

                    wf = add_modify_incar(wf)
                    wf = set_execution_options(wf, category=category, fworker_name=fworker, fw_name_constraint=wf.fws[
                        0].name)
                    wf = set_execution_options(wf, category=category, fworker_name=fworker, fw_name_constraint=wf.fws[
                        1].name)
                    wf = set_execution_options(wf, category="zfs", fworker_name=fworker,
                                               fw_name_constraint=wf.fws[2].name)
                    wf = preserve_fworker(wf)
                    wf.name = wf_name+":biaxial[{}]".format(biaxial_ratio)
                    wfs.append(wf)
    return wfs, category




def cubic_defect(structure, strain, strain_comp, defect_type="substitutions", distort=0.0): #1e-4
    wfs = []
    pc = structure.copy()
    area = 1
    geo_spec = {area*pc.num_sites: [20]}

    defect = ChargedDefectsStructures(pc, antisites_flag=True).defects
    cation, anion = find_cation_anion(pc)

    for sub in range(len(defect[defect_type])):
        print(cation, anion)
        # cation vacancy
        #"as_1_{}_on_{}"
        #"vac_1_{}"
        print(defect[defect_type][sub]["name"])
        if "as_1_{}_on_{}".format(cation, anion) not in defect[defect_type][sub]["name"]:
            continue
        for na, thicks in geo_spec.items():
            for thick in thicks:
                gen_defect = GenDefect(
                    orig_st=pc,
                    defect_type=(defect_type, sub),
                    natom=na,
                    vacuum_thickness=thick,
                    distort=distort,
                    sub_on_side=None,
                    standardize_st=False,
                )
                vasptodb_dict = {}
                vasptodb_dict.update(
                    {
                        "NN": gen_defect.NN,
                        "defect_entry": gen_defect.defect_entry,
                        "lattice_constant": "HSE",
                        "strain_ratio": strain,
                        "strain_component": strain_comp,
                    },
                )

                wf = get_wf_full_hse(
                    structure=gen_defect.defect_st,
                    task_arg=dict(lcharg=True),
                    charge_states=[0],
                    gamma_only=False,
                    gamma_mesh=True,
                    nupdowns=[2],
                    task="hse_relax-hse_scf",
                    vasptodb=vasptodb_dict,
                    wf_addition_name="{}:{}".format(na, thick),
                    double_relax_ediffg=-0.02
                )

                fws = wf.fws
                pyzfsfw = PyzfsFW(parents=fws[-1], structure=gen_defect.defect_st)
                fws.append(pyzfsfw)
                wf = Workflow(fws)
                wf = bash_scp_files(
                    wf,
                    dest="/home/tsai/work2/Research/projects/quantum_sensor/calc_data",
                    port=12350,
                    task_name_constraint="VaspToDb"
                )

                wf = add_modify_incar(wf, {"incar_update": {"EDIFF": 1e-4, "EDIFFG": -0.03}},
                                      fw_name_constraint=wf.fws[0].name)

                wf = add_modify_incar(wf)
                wf = set_execution_options(wf, category="cubic_biaxial", fworker_name="nersc",
                                           fw_name_constraint=wf.fws[0].name)
                wf = set_execution_options(wf, category="cubic_biaxial", fworker_name="nersc",
                                           fw_name_constraint=wf.fws[1].name)
                wf = set_execution_options(wf, category="cubic_biaxial_zfs", fworker_name="nersc",
                                           fw_name_constraint=wf.fws[2].name)
                wf = preserve_fworker(wf)
                wf.name = f"{gen_defect.defect_st.formula}:strain[{strain_comp}.{strain}]"
                wfs.append(wf)
    return wfs
def tri_defect(structure, strain, strain_comp, defect_type="substitutions", distort=0.0): #1e-4
    wfs = []
    pc = structure.copy()
    area = 25
    geo_spec = {area*pc.num_sites: [20]}

    defect = ChargedDefectsStructures(pc, antisites_flag=True).defects
    cation, anion = find_cation_anion(pc)

    for sub in range(len(defect[defect_type])):
        print(cation, anion)
        # cation vacancy
        #"as_1_{}_on_{}"
        #"vac_1_{}"
        print(defect[defect_type][sub]["name"])
        if "as_1_{}_on_{}".format(cation, anion) not in defect[defect_type][sub]["name"]:
            continue
        for na, thicks in geo_spec.items():
            for thick in thicks:
                gen_defect = GenDefect(
                    orig_st=pc,
                    defect_type=(defect_type, sub),
                    natom=na,
                    vacuum_thickness=thick,
                    distort=distort,
                    sub_on_side=None,
                    standardize_st=False,
                )
                vasptodb_dict = {}
                vasptodb_dict.update(
                    {
                        "NN": gen_defect.NN,
                        "defect_entry": gen_defect.defect_entry,
                        "lattice_constant": "HSE",
                        "strain_ratio": strain,
                        "strain_component": strain_comp,
                        "lattice_gamma": 120,
                    },
                )

                wf = get_wf_full_hse(
                    structure=gen_defect.defect_st,
                    task_arg=dict(lcharg=True),
                    charge_states=[0],
                    gamma_only=False,
                    gamma_mesh=True,
                    nupdowns=[2],
                    task="hse_relax-hse_scf",
                    vasptodb=vasptodb_dict,
                    wf_addition_name="{}:{}".format(na, thick),
                    double_relax_ediffg=-0.02
                )

                fws = wf.fws
                pyzfsfw = PyzfsFW(parents=fws[-1], structure=gen_defect.defect_st)
                fws.append(pyzfsfw)
                wf = Workflow(fws)
                wf = bash_scp_files(
                    wf,
                    dest="/home/tsai/work2/Research/projects/quantum_sensor/calc_data",
                    port=12350,
                    task_name_constraint="VaspToDb"
                )

                wf = add_modify_incar(wf, {"incar_update": {"EDIFF": 1e-4, "EDIFFG": -0.03}},
                                      fw_name_constraint=wf.fws[0].name)

                wf = add_modify_incar(wf)
                wf = set_execution_options(wf, category="tri_strained", fworker_name="gpu_nersc",
                                           fw_name_constraint=wf.fws[0].name)
                wf = set_execution_options(wf, category="tri_strained", fworker_name="gpu_nersc",
                                           fw_name_constraint=wf.fws[1].name)
                wf = set_execution_options(wf, category="tri_strained_zfs", fworker_name="cpu_nersc",
                                           fw_name_constraint=wf.fws[2].name)
                wf.name = f"{gen_defect.defect_st.formula}:tri:strain[{strain_comp}.{strain}]"
                wfs.append(wf)
    return wfs
def three_dim_defect(structure, strain, strain_comp, defect_type="substitutions", distort=0.0): #1e-4
    wfs = []
    pc = structure.copy()
    area = 25
    geo_spec = {area*pc.num_sites: [20]}

    defect = ChargedDefectsStructures(pc, antisites_flag=True).defects
    cation, anion = find_cation_anion(pc)

    for sub in range(len(defect[defect_type])):
        print(cation, anion)
        # cation vacancy
        #"as_1_{}_on_{}"
        #"vac_1_{}"
        print(defect[defect_type][sub]["name"])
        if "as_1_{}_on_{}".format(cation, anion) not in defect[defect_type][sub]["name"]:
            continue
        for na, thicks in geo_spec.items():
            for thick in thicks:
                gen_defect = GenDefect(
                    orig_st=pc,
                    defect_type=(defect_type, sub),
                    natom=na,
                    vacuum_thickness=thick,
                    distort=distort,
                    sub_on_side=None,
                    standardize_st=False,
                )
                vasptodb_dict = {}
                vasptodb_dict.update(
                    {
                        "NN": gen_defect.NN,
                        "defect_entry": gen_defect.defect_entry,
                        "lattice_constant": "HSE",
                        "strain_ratio": strain,
                        "strain_component": strain_comp,
                        "lattice_gamma": 120,
                    },
                )

                wf = get_wf_full_hse(
                    structure=gen_defect.defect_st,
                    task_arg=dict(lcharg=True),
                    charge_states=[0],
                    gamma_only=False,
                    gamma_mesh=True,
                    nupdowns=[2],
                    task="hse_relax-hse_scf",
                    vasptodb=vasptodb_dict,
                    wf_addition_name="{}:{}".format(na, thick),
                    double_relax_ediffg=-0.02
                )

                fws = wf.fws
                pyzfsfw = PyzfsFW(parents=fws[-1], structure=gen_defect.defect_st)
                fws.append(pyzfsfw)
                wf = Workflow(fws)
                wf = bash_scp_files(
                    wf,
                    dest="/home/tsai/work2/Research/projects/quantum_sensor/calc_data",
                    port=12350,
                    task_name_constraint="VaspToDb"
                )

                wf = add_modify_incar(wf, {"incar_update": {"EDIFF": 1e-4, "EDIFFG": -0.03}},
                                      fw_name_constraint=wf.fws[0].name)

                wf = add_modify_incar(wf)
                wf = set_execution_options(wf, category="three_dim_strained", fworker_name="nersc",
                                           fw_name_constraint=wf.fws[0].name)
                wf = set_execution_options(wf, category="three_dim_strained", fworker_name="nersc",
                                           fw_name_constraint=wf.fws[1].name)
                wf = set_execution_options(wf, category="three_dim_strained_zfs", fworker_name="nersc",
                                           fw_name_constraint=wf.fws[2].name)
                wf = preserve_fworker(wf)
                wf.name = f"{gen_defect.defect_st.formula}:3D:strain[{strain_comp}.{strain}]"
                wfs.append(wf)
    return wfs


class Run:
    @classmethod
    def std_defect(cls):
        for biaxial in np.arange(-0.05, 0.075, 0.025):
            if biaxial == 0:
                continue
            wfs, cat = standard_defect(distort=0, biaxial_ratio=round(biaxial, 3), fworker="efrc")
            for wf in wfs:
                LPAD = LaunchPad.from_file('wf/atomate_config/my_launchpad.yaml')
                # LPAD.add_wf(wf)

    @classmethod
    def cubic_defect(cls):
        for json_f in glob.glob("structure/cubic_sc/strained_pristine/xx_yy/Mo30Se60*"):
            mx2 = loadfn(json_f)
            strains = mx2['strain']
            sts = mx2['st']
            print(f"strain: {len(strains)}")
            print(f"st: {len(sts)}")
            for st, strain in zip(sts, strains):
                if strain == 0:
                    continue
                wfs = cubic_defect(st, strain, strain_comp="xx_yy", defect_type="substitutions", distort=0.0)
                for wf in wfs:
                    LPAD = LaunchPad.from_file('wf/atomate_config/my_launchpad.yaml')
                    LPAD.add_wf(wf)
            break

    @classmethod
    def tri_defect(cls):
        for strain_comp in ["xx_yy"]:
            for chemical in ["W1Se2", "W1S2"]:
                for json_f in glob.glob(f"structure/pc/strained_pc/{strain_comp}/-0.008:0.008/{chemical}.json"):
                    mx2 = loadfn(json_f)
                    strains = mx2['strain']
                    sts = mx2['st']
                    print(f"strain: {len(strains)}")
                    print(f"st: {len(sts)}")
                    for st, strain in zip(sts, strains):
                        if strain == 0:
                            continue
                        wfs = tri_defect(st, strain, strain_comp=strain_comp, defect_type="substitutions", distort=0.0)
                        for wf in wfs:
                            LPAD = LaunchPad.from_file('wf/atomate_config/my_launchpad.yaml')
                            LPAD.add_wf(wf)
                    break
    @classmethod
    def three_dim_defect(cls):
        for json_f in glob.glob("structure/pc/strained_pc/3D_TMDs/zz/0.3:0.6/Mo2S4.json"):
            mx2 = loadfn(json_f)
            strains = mx2['strain']
            sts = mx2['st']
            print(f"strain: {len(strains)}")
            print(f"st: {len(sts)}")
            for st, strain in zip(sts, strains):
                # if strain == 0:
                #     continue
                wfs = three_dim_defect(st, strain, strain_comp="zz", defect_type="substitutions", distort=0.0)
                for wf in wfs:
                    LPAD = LaunchPad.from_file('wf/atomate_config/my_launchpad.yaml')
                    LPAD.add_wf(wf)
            break


if __name__ == '__main__':
    Run.tri_defect()