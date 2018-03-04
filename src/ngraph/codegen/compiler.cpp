/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <iostream>

#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/CodeGen/ObjectFilePCHContainerOperations.h>
#include <clang/Driver/DriverDiagnostic.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/FrontendDiagnostic.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/FrontendTool/Utils.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/ExecutionEngine/MCJIT.h> // forces JIT to link in
#include <llvm/IR/Module.h>
#include <llvm/LinkAllPasses.h>
#include <llvm/Option/Arg.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Timer.h>
#include <llvm/Support/raw_ostream.h>

#include "header_resource.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

#if defined(__clang__)
#define IS_RTTI_ENABLED __has_feature(cxx_rtti)
#elif defined(__GNUC__)
#define IS_RTTI_ENABLED __GXX_RTTI
#else
// Unknown compiler so assume RTTI is enabled by default
#define IS_RTTI_ENABLED 1
#endif

#if IS_RTTI_ENABLED
#error "This source file interfaces with LLVM and Clang and must be compiled with RTTI disabled"
#endif

#define USE_BUILTIN

using namespace clang;
using namespace llvm;
using namespace std;
using namespace ngraph;

unordered_map<string, string> s_pch_cache;

class StaticDtor
{
public:
    ~StaticDtor()
    {
        for (const pair<string, string>& pch_info : s_pch_cache)
        {
            file_util::remove_file(pch_info.second);
        }
    }
} s_static_dtor;

codegen::Module::Module(unique_ptr<llvm::Module>& module)
    : m_module(move(module))
{
}

codegen::Module::~Module()
{
}

unique_ptr<llvm::Module> codegen::Module::take_module()
{
    return move(m_module);
}

codegen::Compiler::Compiler()
{
}

codegen::Compiler::~Compiler()
{
}

void codegen::Compiler::set_precompiled_header_source(const string& source)
{
    m_precomiled_header_source = source;
}

void codegen::Compiler::add_header_search_path(const string& path)
{
    m_extra_header_search_paths.push_back(path);
}

unique_ptr<codegen::Module> codegen::Compiler::compile(const string& source)
{
    static stopwatch timer;
    timer.start();
    m_source_name = "code.cpp";
    static bool llvm_initialized = false;
    if (!llvm_initialized)
    {
        llvm_initialized = true;
        InitializeNativeTarget();
        LLVMInitializeNativeAsmPrinter();
        LLVMInitializeNativeAsmParser();
    }

    // Prepare compilation arguments
    vector<const char*> args;
    args.push_back(m_source_name.c_str());

    // Inlining thresholds are forced to a very high value
    // to ensure all Eigen code gets properly inlined
    // This is for both Eigen strong and weak inlines
    args.push_back("-mllvm");
    args.push_back("-inline-threshold=1000000");

    // Prepare DiagnosticEngine
    IntrusiveRefCntPtr<DiagnosticOptions> diag_options = new DiagnosticOptions();
    diag_options->ErrorLimit = 20;
    diag_options->ShowCarets = false;
    diag_options->ShowFixits = false;
    IntrusiveRefCntPtr<DiagnosticIDs> diag_id(new DiagnosticIDs());
    DiagnosticsEngine diag_engine(diag_id, &*diag_options);

    // Create and initialize CompilerInstance
    m_compiler_instance = unique_ptr<CompilerInstance>(new CompilerInstance());
    DiagnosticConsumer* diag_consumer;
    if (m_enable_diag_output)
    {
        diag_consumer = new TextDiagnosticPrinter(errs(), &*diag_options);
    }
    else
    {
        diag_consumer = new IgnoringDiagConsumer();
    }
    m_compiler_instance->createDiagnostics(diag_consumer);

    // Initialize CompilerInvocation
    CompilerInvocation::CreateFromArgs(
        m_compiler_instance->getInvocation(), &args[0], &args[0] + args.size(), diag_engine);

    configure_search_path();

    // Language options
    // These are the C++ features needed to compile ngraph headers
    // and any dependencies like Eigen
    auto language_options = m_compiler_instance->getInvocation().getLangOpts();
    language_options->CPlusPlus = 1;
    language_options->CPlusPlus11 = 1;
    language_options->Bool = 1;
    language_options->Exceptions = 1;
    language_options->CXXExceptions = 1;
    language_options->WChar = 1;
    language_options->RTTI = 1;
    // Enable OpenMP for Eigen
    language_options->OpenMP = 1;
    language_options->OpenMPUseTLS = 1;

    // CodeGen options
    auto& codegen_options = m_compiler_instance->getInvocation().getCodeGenOpts();
    codegen_options.OptimizationLevel = 3;
    codegen_options.RelocationModel = "static";
    // codegen_options.CodeModel = "medium";
    codegen_options.ThreadModel = "posix";
    codegen_options.FloatABI = "hard";
    codegen_options.OmitLeafFramePointer = 1;
    codegen_options.VectorizeLoop = 1;
    codegen_options.VectorizeSLP = 1;
    codegen_options.CXAAtExit = 1;

    if (m_debuginfo_enabled)
    {
        codegen_options.setDebugInfo(codegenoptions::FullDebugInfo);
    }

    // Enable various target features
    auto& target_options = m_compiler_instance->getInvocation().getTargetOpts();
    target_options.CPU = sys::getHostCPUName();

    unique_ptr<codegen::Module> result;
    PreprocessorOptions& preprocessor_options =
        m_compiler_instance->getInvocation().getPreprocessorOpts();

    // Clear warnings and errors
    m_compiler_instance->getDiagnosticClient().clear();

    preprocessor_options.RetainRemappedFileBuffers = true;

    auto pch_it = s_pch_cache.find(m_precomiled_header_source);
    if (pch_it != s_pch_cache.end())
    {
        preprocessor_options.ImplicitPCHInclude = pch_it->second;
    }
    else
    {
        string pch_file = generate_pch(m_precomiled_header_source);
        s_pch_cache.insert({m_precomiled_header_source, pch_file});
        preprocessor_options.ImplicitPCHInclude = pch_file;
    }
    preprocessor_options.DisablePCHValidation = 1;

    // Map code filename to a memoryBuffer
    StringRef source_ref(source);
    unique_ptr<MemoryBuffer> buffer = MemoryBuffer::getMemBufferCopy(source_ref);
    preprocessor_options.RemappedFileBuffers.push_back({m_source_name, buffer.get()});

    // Create and execute action
    unique_ptr<llvm::Module> rc;
    bool reinitialize = false;
    m_action.reset(new EmitCodeGenOnlyAction());
    if (m_compiler_instance->ExecuteAction(*m_action) == true)
    {
        rc = m_action->takeModule();
    }
    else
    {
        reinitialize = true;
    }

    buffer.release();

    preprocessor_options.RemappedFileBuffers.pop_back();

    if (rc)
    {
        result = move(unique_ptr<codegen::Module>(new codegen::Module(rc)));
    }
    else
    {
        result = move(unique_ptr<codegen::Module>(nullptr));
    }

    timer.stop();
    NGRAPH_INFO << timer.get_milliseconds() << ", " << timer.get_total_milliseconds();

    return result;
}

std::string codegen::Compiler::generate_pch(const string& source)
{
    PreprocessorOptions& preprocessor_options =
        m_compiler_instance->getInvocation().getPreprocessorOpts();
    string pch_path = file_util::tmp_filename();
    m_compiler_instance->getFrontendOpts().OutputFile = pch_path;

    // Map code filename to a memoryBuffer
    StringRef source_ref(source);
    unique_ptr<MemoryBuffer> buffer = MemoryBuffer::getMemBufferCopy(source_ref);
    preprocessor_options.RemappedFileBuffers.push_back({m_source_name, buffer.get()});

    // Create and execute action
    clang::GeneratePCHAction* compilerAction = new clang::GeneratePCHAction();
    if (m_compiler_instance->ExecuteAction(*compilerAction) == true)
    {
        // NGRAPH_INFO << "success"
        //             << "\n"
        //             << source;
    }

    buffer.release();
    preprocessor_options.RemappedFileBuffers.pop_back();

    delete compilerAction;

    return pch_path;
}

void codegen::Compiler::configure_search_path()
{
#ifdef USE_BUILTIN
    load_headers_from_resource();
#elif defined(__APPLE__)
    add_header_search_path(EIGEN_HEADERS_PATH);
    add_header_search_path(MKLDNN_HEADERS_PATH);
    add_header_search_path(TBB_HEADERS_PATH);
    add_header_search_path(NGRAPH_HEADERS_PATH);
    add_header_search_path(INSTALLED_HEADERS_PATH);
    add_header_search_path(CLANG_BUILTIN_HEADERS_PATH);

    add_header_search_path("/Library/Developer/CommandLineTools/usr/include/c++/v1");
#else
    // Add base toolchain-supplied header paths
    // Ideally one would use the Linux toolchain definition in clang/lib/Driver/ToolChains.h
    // But that's a private header and isn't part of the public libclang API
    // Instead of re-implementing all of that functionality in a custom toolchain
    // just hardcode the paths relevant to frequently used build/test machines for now
    add_header_search_path(CLANG_BUILTIN_HEADERS_PATH);
    add_header_search_path("/usr/include/x86_64-linux-gnu");
    add_header_search_path("/usr/include");

    // Search for headers in
    //    /usr/include/x86_64-linux-gnu/c++/N.N
    //    /usr/include/c++/N.N
    // and add them to the header search path

    file_util::iterate_files("/usr/include/x86_64-linux-gnu/c++/",
                             [&](const string& file, bool is_dir) {
                                 if (is_dir)
                                 {
                                     string dir_name = file_util::get_file_name(file);
                                     if (is_version_number(dir_name))
                                     {
                                         add_header_search_path(file);
                                     }
                                 }
                             });

    file_util::iterate_files("/usr/include/c++/", [&](const string& file, bool is_dir) {
        if (is_dir)
        {
            string dir_name = file_util::get_file_name(file);
            if (is_version_number(dir_name))
            {
                add_header_search_path(file);
            }
        }
    });

    add_header_search_path(EIGEN_HEADERS_PATH);
    add_header_search_path(MKLDNN_HEADERS_PATH);
    add_header_search_path(TBB_HEADERS_PATH);
    add_header_search_path(NGRAPH_HEADERS_PATH);
    add_header_search_path(INSTALLED_HEADERS_PATH);
#endif

#ifdef CUDA_HEADER_PATHS
    // Only needed for GPU backend
    add_header_search_path(CUDA_HEADER_PATHS);
#endif

#ifdef NGRAPH_DISTRIBUTED
    add_header_search_path(MPI_HEADER_PATH);
#endif
}

void codegen::Compiler::load_headers_from_resource()
{
    const string builtin_root = "/$builtin";
    HeaderSearchOptions& hso = m_compiler_instance->getInvocation().getHeaderSearchOpts();
    PreprocessorOptions& preprocessor_options =
        m_compiler_instance->getInvocation().getPreprocessorOpts();
    for (const string& search_path : builtin_search_paths)
    {
        string builtin = builtin_root + search_path;
        hso.AddPath(builtin, clang::frontend::System, false, false);
    }
    for (const string& search_path : m_extra_header_search_paths)
    {
        hso.AddPath(search_path, clang::frontend::System, false, false);
    }
    for (const pair<string, string>& header_info : builtin_headers)
    {
        string absolute_path = header_info.first;
        string builtin = builtin_root + absolute_path;
        std::unique_ptr<llvm::MemoryBuffer> mb(
            llvm::MemoryBuffer::getMemBuffer(header_info.second, builtin));
        preprocessor_options.addRemappedFile(builtin, mb.release());
    }
}
